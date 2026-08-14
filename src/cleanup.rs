use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, mpsc};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use regex::{Captures, Regex};

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[\w']+\b").unwrap());
static INITIAL_I_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(\s*)i((?:\s+(?:mean|think|guess|believe|know|want|need|will|would|can|could|should|am|was|have|had|do|did|feel|see|understand|don't|dont|can't|cant|won't|wont|wouldn't|wouldnt|shouldn't|shouldnt)\b|'(?:m|ve|ll|d)\b|$))",
    )
    .unwrap()
});
static DECIMAL_RE: LazyLock<Regex> = LazyLock::new(|| number_regex(true));
static NUMERIC_RE: LazyLock<Regex> = LazyLock::new(|| number_regex(false));

#[derive(Default)]
struct Glossary {
    always: Vec<(String, String)>,
    likely: Vec<(String, String)>,
    contextual: Vec<(String, String)>,
    terms: Vec<String>,
    legacy: Option<String>,
}

pub struct Options {
    pub model_enabled: bool,
    pub timeout: Duration,
    pub glossary_file: Option<PathBuf>,
}

pub fn process(text: &str, options: &Options) -> String {
    let raw_word_count = word_count(text);
    let glossary = match load_glossary(options.glossary_file.as_deref()) {
        Ok(glossary) => glossary,
        Err(error) => {
            eprintln!("Warning: {error:#}; using local cleanup without glossary.");
            Glossary::default()
        }
    };
    let prepared = apply_guaranteed_corrections(&normalize_spoken_numerics(text), &glossary.always);
    let local = normalize_short_statement_style(&prepared);
    let should_use_model =
        baml_sdk::should_clean_with_model(raw_word_count as i64, options.model_enabled)
            .unwrap_or(false);
    if !should_use_model {
        return local;
    }

    let transcript = prepared.clone();
    let prompt_glossary = glossary.prompt_text();
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = sender.send(baml_sdk::clean_transcript(transcript, prompt_glossary));
    });

    let result = match receiver.recv_timeout(options.timeout) {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            eprintln!("Warning: transcript post-processing failed: {error}; using local cleanup.");
            return local;
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "Warning: transcript post-processing timed out after {:.1}s; using local cleanup.",
                options.timeout.as_secs_f64()
            );
            return local;
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            eprintln!(
                "Warning: transcript post-processing stopped unexpectedly; using local cleanup."
            );
            return local;
        }
    };
    let Some(cleaned) = result.text.filter(|text| !text.trim().is_empty()) else {
        let detail = result
            .error
            .unwrap_or_else(|| "empty model output".to_owned());
        eprintln!("Warning: transcript post-processing failed: {detail}; using local cleanup.");
        return local;
    };
    if looks_like_unwanted_non_latin_translation(&prepared, &cleaned) {
        eprintln!("Warning: transcript cleanup changed the language; using local cleanup.");
        return local;
    }
    normalize_final_transcript(&apply_guaranteed_corrections(&cleaned, &glossary.always))
}

fn number_regex(decimal: bool) -> Regex {
    let digit = "zero|oh|one|two|three|four|five|six|seven|eight|nine";
    let teen = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen";
    let tens = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety";
    let base = format!("(?:{tens})(?:[\\s-]+(?:{digit}))?|(?:{teen})|(?:{digit})");
    let number =
        format!("(?:{digit})[\\s-]+hundred(?:[\\s-]+and)?(?:[\\s-]+(?:{base}))?|(?:{base})");
    let pattern = if decimal {
        format!(
            "(?i)\\b(?P<integer>{number})[\\s-]+point[\\s-]+(?P<fraction>(?:{digit})(?:[\\s-]+(?:{digit}))*)\\b"
        )
    } else {
        format!("(?i)\\bnumeric[\\s-]+(?P<number>{number})\\b")
    };
    Regex::new(&pattern).unwrap()
}

fn load_glossary(path: Option<&Path>) -> Result<Glossary> {
    match path {
        Some(path) => {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("could not read glossary {}", path.display()))?;
            parse_glossary(&raw)
        }
        None => Ok(Glossary::default()),
    }
}

fn parse_glossary(raw: &str) -> Result<Glossary> {
    let has_sections = raw.lines().any(|line| {
        let line = line.trim();
        line.starts_with('[') && line.ends_with(']')
    });
    if !has_sections {
        return Ok(Glossary {
            legacy: (!raw.trim().is_empty()).then(|| raw.trim().to_owned()),
            ..Glossary::default()
        });
    }

    let mut sections: HashMap<String, Vec<(usize, String)>> =
        ["always", "likely", "contextual", "terms"]
            .into_iter()
            .map(|name| (name.to_owned(), Vec::new()))
            .collect();
    let mut current: Option<String> = None;
    for (index, original) in raw.lines().enumerate() {
        let line_number = index + 1;
        let line = original.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let section = line[1..line.len() - 1].trim().to_ascii_lowercase();
            if !sections.contains_key(&section) {
                bail!("unknown glossary section [{section}] on line {line_number}")
            }
            current = Some(section);
            continue;
        }
        let section = current.as_ref().with_context(|| {
            format!("glossary entry appears before a section on line {line_number}")
        })?;
        sections
            .get_mut(section)
            .unwrap()
            .push((line_number, line.to_owned()));
    }

    let mut seen = HashSet::new();
    let mut glossary = Glossary::default();
    for section in ["always", "likely", "contextual"] {
        let mut rules = Vec::new();
        for (line_number, line) in sections.remove(section).unwrap() {
            let Some((source, replacement)) = line.split_once("->") else {
                bail!(
                    "glossary [{section}] entry on line {line_number} must use 'source -> replacement'"
                )
            };
            let source = source.trim();
            let replacement = replacement.trim();
            if source.is_empty() || replacement.is_empty() {
                bail!(
                    "glossary [{section}] entry on line {line_number} has an empty source or replacement"
                )
            }
            if !seen.insert(source.to_lowercase()) {
                bail!("glossary source {source:?} appears in more than one section")
            }
            rules.push((source.to_owned(), replacement.to_owned()));
        }
        match section {
            "always" => glossary.always = rules,
            "likely" => glossary.likely = rules,
            "contextual" => glossary.contextual = rules,
            _ => unreachable!(),
        }
    }
    glossary.terms = sections
        .remove("terms")
        .unwrap()
        .into_iter()
        .map(|(line_number, term)| {
            if term.contains("->") {
                bail!("glossary [terms] entry on line {line_number} must be a term")
            }
            Ok(term)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(glossary)
}

impl Glossary {
    fn prompt_text(&self) -> String {
        if let Some(legacy) = &self.legacy {
            return xml_escape(legacy);
        }
        let mut parts = Vec::new();
        append_rules(&mut parts, "always", &self.always);
        append_rules(&mut parts, "likely", &self.likely);
        append_rules(&mut parts, "contextual", &self.contextual);
        if !self.terms.is_empty() {
            parts.push("<canonical_terms>".to_owned());
            parts.extend(self.terms.iter().map(|term| xml_escape(term)));
            parts.push("</canonical_terms>".to_owned());
        }
        parts.join("\n")
    }
}

fn append_rules(parts: &mut Vec<String>, name: &str, rules: &[(String, String)]) {
    if rules.is_empty() {
        return;
    }
    parts.push(format!("<{name}>"));
    parts.extend(rules.iter().map(|(source, replacement)| {
        format!("{} => {}", xml_escape(source), xml_escape(replacement))
    }));
    parts.push(format!("</{name}>"));
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn normalize_spoken_numerics(text: &str) -> String {
    let decimals = DECIMAL_RE.replace_all(text, |captures: &Captures<'_>| {
        let Some(integer) = parse_spoken_number(&captures["integer"]) else {
            return captures[0].to_owned();
        };
        let fraction = split_number_words(&captures["fraction"])
            .into_iter()
            .map(digit_value)
            .collect::<Option<Vec<_>>>();
        match fraction {
            Some(digits) => format!(
                "{integer}.{}",
                digits
                    .into_iter()
                    .map(|number| number.to_string())
                    .collect::<String>()
            ),
            None => captures[0].to_owned(),
        }
    });
    NUMERIC_RE
        .replace_all(&decimals, |captures: &Captures<'_>| {
            parse_spoken_number(&captures["number"])
                .map(|number| number.to_string())
                .unwrap_or_else(|| captures[0].to_owned())
        })
        .into_owned()
}

fn parse_spoken_number(text: &str) -> Option<u32> {
    let words = split_number_words(text);
    let mut current = 0;
    let mut saw_number = false;
    let mut previous = "";
    for (index, word) in words.iter().enumerate() {
        if word.eq_ignore_ascii_case("and") {
            if previous != "hundred" || index == words.len() - 1 {
                return None;
            }
            previous = "and";
        } else if let Some(number) = digit_value(word) {
            if matches!(previous, "digit" | "teen") {
                return None;
            }
            current += number;
            saw_number = true;
            previous = "digit";
        } else if let Some(number) = teen_value(word) {
            if matches!(previous, "digit" | "teen" | "tens") {
                return None;
            }
            current += number;
            saw_number = true;
            previous = "teen";
        } else if let Some(number) = tens_value(word) {
            if matches!(previous, "digit" | "teen" | "tens") {
                return None;
            }
            current += number;
            saw_number = true;
            previous = "tens";
        } else if word.eq_ignore_ascii_case("hundred") && saw_number {
            if previous != "digit" {
                return None;
            }
            current *= 100;
            previous = "hundred";
        } else {
            return None;
        }
    }
    saw_number.then_some(current)
}

fn split_number_words(text: &str) -> Vec<&str> {
    text.split([' ', '\t', '\n', '-'])
        .filter(|word| !word.is_empty())
        .collect()
}

fn digit_value(word: &str) -> Option<u32> {
    Some(match word.to_ascii_lowercase().as_str() {
        "zero" | "oh" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        _ => return None,
    })
}

fn teen_value(word: &str) -> Option<u32> {
    Some(match word.to_ascii_lowercase().as_str() {
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        _ => return None,
    })
}

fn tens_value(word: &str) -> Option<u32> {
    Some(match word.to_ascii_lowercase().as_str() {
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return None,
    })
}

fn apply_guaranteed_corrections(text: &str, rules: &[(String, String)]) -> String {
    let mut rules = rules.to_vec();
    rules.sort_by_key(|(source, _)| std::cmp::Reverse(source.len()));
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < text.len() {
        let rule = rules.iter().find(|(source, _)| {
            let end = index + source.len();
            end <= text.len()
                && text.is_char_boundary(end)
                && text[index..end].eq_ignore_ascii_case(source)
                && is_boundary_before(text, index)
                && is_boundary_after(text, end)
        });
        if let Some((source, replacement)) = rule {
            output.push_str(replacement);
            index += source.len();
        } else {
            let character = text[index..].chars().next().unwrap();
            output.push(character);
            index += character.len_utf8();
        }
    }
    output
}

fn is_boundary_before(text: &str, index: usize) -> bool {
    index == 0
        || !text[..index]
            .chars()
            .next_back()
            .is_some_and(is_word_character)
}

fn is_boundary_after(text: &str, index: usize) -> bool {
    index == text.len() || !text[index..].chars().next().is_some_and(is_word_character)
}

fn is_word_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

fn normalize_final_transcript(text: &str) -> String {
    normalize_short_statement_style(&normalize_spoken_numerics(text))
}

fn normalize_short_statement_style(text: &str) -> String {
    if text
        .chars()
        .any(|character| character.is_alphabetic() && !character.is_ascii_alphabetic())
        || text.contains('?')
        || sentence_end_count(text) >= 2
    {
        return text.to_owned();
    }
    if word_count(text) > 10 {
        return normalize_long_statement_style(text);
    }

    let (body, suffix) = split_trailing_whitespace(text);
    let body = body.strip_suffix('.').unwrap_or(body).trim_end();
    let body = INITIAL_I_RE.replace(body, "${1}I${2}");
    let initial_a = Regex::new(r"^(\s*)A\b").unwrap();
    let body = initial_a.replace(&body, "${1}a");
    let initial_word = Regex::new(r"^(\s*)([A-Z][a-z]+)(\b|')").unwrap();
    let body = initial_word.replace(&body, |captures: &Captures<'_>| {
        format!(
            "{}{}{}",
            &captures[1],
            captures[2].to_lowercase(),
            &captures[3]
        )
    });
    format!("{body}{suffix}")
}

fn normalize_long_statement_style(text: &str) -> String {
    let (body, suffix) = split_trailing_whitespace(text);
    let body = INITIAL_I_RE.replace(body, "${1}I${2}");
    let initial_word = Regex::new(r"^(\s*)([a-z]+)(\b|')").unwrap();
    let mut body = initial_word
        .replace(&body, |captures: &Captures<'_>| {
            let mut word = captures[2].to_owned();
            word[0..1].make_ascii_uppercase();
            format!("{}{word}{}", &captures[1], &captures[3])
        })
        .into_owned();
    if !body.is_empty() && !body.ends_with(['.', '!', '?']) {
        body.push('.');
    }
    format!("{body}{suffix}")
}

fn split_trailing_whitespace(text: &str) -> (&str, &str) {
    let body = text.trim_end();
    (body, &text[body.len()..])
}

fn word_count(text: &str) -> usize {
    WORD_RE.find_iter(text).count()
}

fn sentence_end_count(text: &str) -> usize {
    let characters: Vec<_> = text.chars().collect();
    characters
        .iter()
        .enumerate()
        .filter(|(index, character)| match character {
            '!' | '?' => true,
            '.' => {
                let previous_digit = index
                    .checked_sub(1)
                    .and_then(|i| characters.get(i))
                    .is_some_and(|c| c.is_ascii_digit());
                let next_digit = characters
                    .get(index + 1)
                    .is_some_and(|c| c.is_ascii_digit());
                !(previous_digit && next_digit)
            }
            _ => false,
        })
        .count()
}

fn looks_like_unwanted_non_latin_translation(source: &str, processed: &str) -> bool {
    let (source_latin, source_non_latin) = script_counts(source);
    let (processed_latin, processed_non_latin) = script_counts(processed);
    let allowed_growth = 6.max(source_non_latin * 2);
    source_latin > 0
        && processed_non_latin > 0
        && processed_non_latin > processed_latin
        && processed_non_latin > source_non_latin + allowed_growth
}

fn script_counts(text: &str) -> (usize, usize) {
    text.chars()
        .filter(|character| character.is_alphabetic())
        .fold((0, 0), |(latin, non_latin), character| {
            if character.is_ascii_alphabetic() {
                (latin + 1, non_latin)
            } else {
                (latin, non_latin + 1)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_spoken_numbers() {
        assert_eq!(normalize_spoken_numerics("zero point one"), "0.1");
        assert_eq!(
            normalize_spoken_numerics("version twelve point zero"),
            "version 12.0"
        );
        assert_eq!(
            normalize_spoken_numerics("one hundred and five point six"),
            "105.6"
        );
        assert_eq!(normalize_spoken_numerics("numeric twenty one"), "21");
        assert_eq!(
            normalize_spoken_numerics("one and two point three"),
            "one and 2.3"
        );
    }

    #[test]
    fn guaranteed_rules_are_boundary_aware_and_do_not_cascade() {
        let rules = vec![
            ("code".to_owned(), "Codex".to_owned()),
            ("cloud code".to_owned(), "Claude Code".to_owned()),
            ("cat".to_owned(), "dog".to_owned()),
        ];
        assert_eq!(
            apply_guaranteed_corrections("Cloud code and cat scatter", &rules),
            "Claude Code and dog scatter"
        );
    }

    #[test]
    fn preserves_established_statement_style() {
        assert_eq!(normalize_final_transcript("Fair point."), "fair point");
        assert_eq!(
            normalize_final_transcript("Because it will be simpler this way."),
            "because it will be simpler this way"
        );
        assert_eq!(
            normalize_final_transcript("Version zero point one."),
            "version 0.1"
        );
        assert_eq!(normalize_final_transcript("A fair point."), "a fair point");
        assert_eq!(normalize_final_transcript("i mean"), "I mean");
        assert_eq!(normalize_final_transcript("i'm sure"), "I'm sure");
        assert_eq!(normalize_final_transcript("It's fine."), "it's fine");
        assert_eq!(normalize_final_transcript("API request."), "API request");
        assert_eq!(normalize_final_transcript("Use API."), "use API");
        assert_eq!(
            normalize_final_transcript("for i in items"),
            "for i in items"
        );
        assert_eq!(
            normalize_final_transcript("TypeScript type."),
            "TypeScript type"
        );
        assert_eq!(
            normalize_final_transcript("How can we solve it?"),
            "How can we solve it?"
        );
        assert_eq!(
            normalize_final_transcript("That's a fair point. Let's go with this approach."),
            "That's a fair point. Let's go with this approach."
        );
        assert_eq!(
            normalize_final_transcript("Хорошая мысль."),
            "Хорошая мысль."
        );
        assert_eq!(
            normalize_final_transcript(
                "because it will be simpler this way and it reduces complexity overall"
            ),
            "Because it will be simpler this way and it reduces complexity overall."
        );
        assert_eq!(
            normalize_final_transcript(
                "i think this approach will be simpler because it reduces complexity overall"
            ),
            "I think this approach will be simpler because it reduces complexity overall."
        );
        assert_eq!(
            normalize_final_transcript(
                "TypeScript type inference should stay unchanged when it starts the statement"
            ),
            "TypeScript type inference should stay unchanged when it starts the statement."
        );
    }

    #[test]
    fn parses_the_system_glossary_shape() {
        let glossary = parse_glossary(
            "[always]\nengine x -> nginx\n[likely]\ncloud code -> Claude Code\n[contextual]\ncodecs -> Codex\n[terms]\nTypeScript\n",
        )
        .unwrap();
        assert_eq!(
            glossary.always[0],
            ("engine x".to_owned(), "nginx".to_owned())
        );
        assert!(
            glossary
                .prompt_text()
                .contains("<canonical_terms>\nTypeScript")
        );
    }

    #[test]
    fn rejects_duplicate_glossary_sources() {
        assert!(
            parse_glossary("[always]\ncodecs -> Codex\n[contextual]\ncodecs -> Codex").is_err()
        );
    }
}
