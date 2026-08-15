class GlossaryRule {
    source: string,
    replacement: string,
}

class Glossary {
    always: GlossaryRule[],
    likely: GlossaryRule[],
    contextual: GlossaryRule[],
    terms: string[],
    legacy: string?,
}

enum GlossarySection {
    Always,
    Likely,
    Contextual,
    Terms,
}

class WordSpan {
    text: string,
    start: int,
    end: int,
}

class NumericMatch {
    end_word: int,
    replacement: string,
}

class ModelCleanAttempt {
    result: CleanResult?,
    timed_out: bool,
}

function empty_glossary() -> Glossary {
    Glossary { always: [], likely: [], contextual: [], terms: [], legacy: null }
}

function glossary_section(name: string, line_number: int) -> GlossarySection {
    match (name.to_lower_case()) {
        "always" => GlossarySection.Always,
        "likely" => GlossarySection.Likely,
        "contextual" => GlossarySection.Contextual,
        "terms" => GlossarySection.Terms,
        _ => invalid_argument(`unknown glossary section [${name}] on line ${line_number}`),
    }
}

function parse_glossary(raw: string) -> Glossary {
    let has_sections = raw.lines().some((original) -> {
        let line = original.trim();
        line.starts_with("[") && line.ends_with("]")
    });
    if (!has_sections) {
        let value = raw.trim();
        let glossary = empty_glossary();
        glossary.legacy
            = if (value == "") {
                null
            } else {
                value
            };
        return glossary;
    }

    let glossary = empty_glossary();
    let seen_sources: map<string, bool> = {};
    let current: GlossarySection? = null;
    let lines = raw.lines();
    let index = 0;
    while (index < lines.length()) {
        let line_number = index + 1;
        let line = lines[index].trim();
        index += 1;
        if (line == "" || line.starts_with("#")) {
            continue;
        }
        if (line.starts_with("[") && line.ends_with("]")) {
            current = glossary_section(line.substring(1, line.length() - 1).trim(), line_number);
            continue;
        }
        let section = current;
        if (section == GlossarySection.Terms) {
            if (line.includes("->")) {
                invalid_argument(`glossary [terms] entry on line ${line_number} must be a term`)
            }
            glossary.terms.push(line);
            continue;
        }

        let arrow = line.index_of("->")
            ?? invalid_argument(`glossary entry on line ${line_number} must use 'source -> replacement'`);
        let source = line.substring(0, arrow).trim();
        let replacement = line.substring(arrow + 2, line.length()).trim();
        if (source == "" || replacement == "") {
            invalid_argument(`glossary entry on line ${line_number} has an empty source or replacement`)
        }
        let normalized = source.to_lower_case();
        if (seen_sources.has(normalized)) {
            invalid_argument(`glossary source ${source} appears in more than one section`)
        }
        seen_sources.set(normalized, true);
        let rule = GlossaryRule { source: source, replacement: replacement };
        match (section) {
            GlossarySection.Always => {
                glossary.always.push(rule);
                null
            },
            GlossarySection.Likely => {
                glossary.likely.push(rule);
                null
            },
            GlossarySection.Contextual => {
                glossary.contextual.push(rule);
                null
            },
            GlossarySection.Terms => null,
        };
    }
    glossary
}

function load_glossary(path: string?) -> Glossary {
    if let glossary_path: string = path {
        parse_glossary(baml.fs.read(glossary_path))
    } else {
        empty_glossary()
    }
}

function xml_escape(value: string) -> string {
    value.replace_all("&", "&amp;").replace_all("<", "&lt;").replace_all(">", "&gt;")
}

function append_prompt_rules(parts: string[], name: string, rules: GlossaryRule[]) -> null {
    if (rules.length() == 0) {
        return null;
    }
    parts.push(`<${name}>`);
    for (let rule in rules) {
        parts.push(`${xml_escape(rule.source)} => ${xml_escape(rule.replacement)}`);
    }
    parts.push(`</${name}>`);
    null
}

function glossary_prompt(glossary: Glossary) -> string {
    if let legacy: string = glossary.legacy {
        return xml_escape(legacy);
    }
    let parts: string[] = [];
    append_prompt_rules(parts, "always", glossary.always);
    append_prompt_rules(parts, "likely", glossary.likely);
    append_prompt_rules(parts, "contextual", glossary.contextual);
    if (glossary.terms.length() > 0) {
        parts.push("<canonical_terms>");
        for (let term in glossary.terms) {
            parts.push(xml_escape(term));
        }
        parts.push("</canonical_terms>");
    }
    parts.join("\n")
}

function is_word_character(character: string) -> bool {
    character.is_alphanumeric() || character == "_" || character == "'"
}

function is_boundary_character(character: string) -> bool {
    character.is_alphanumeric() || character == "_"
}

function word_spans(text: string) -> WordSpan[] {
    let spans: WordSpan[] = [];
    let index = 0;
    while (index < text.length()) {
        if (!is_word_character(text.char_at(index))) {
            index += 1;
            continue;
        }
        let start = index;
        while (index < text.length() && is_word_character(text.char_at(index))) {
            index += 1;
        }
        spans.push(WordSpan { text: text.substring(start, index), start: start, end: index });
    }
    spans
}

function word_count(text: string) -> int {
    word_spans(text).length()
}

function digit_value(word: string) -> int? {
    match (word.to_lower_case()) {
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
        _ => null,
    }
}

function teen_value(word: string) -> int? {
    match (word.to_lower_case()) {
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
        _ => null,
    }
}

function tens_value(word: string) -> int? {
    match (word.to_lower_case()) {
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => null,
    }
}

function parse_spoken_number(words: string[]) -> int? {
    let current = 0;
    let saw_number = false;
    let previous = "";
    let index = 0;
    while (index < words.length()) {
        let word = words[index].to_lower_case();
        if (word == "and") {
            if (previous != "hundred" || index == words.length() - 1) {
                return null;
            }
            previous = "and";
        } else if let number: int = digit_value(word) {
            if (["digit", "teen"].includes(previous)) {
                return null;
            }
            current += number;
            saw_number = true;
            previous = "digit";
        } else if let number: int = teen_value(word) {
            if (["digit", "teen", "tens"].includes(previous)) {
                return null;
            }
            current += number;
            saw_number = true;
            previous = "teen";
        } else if let number: int = tens_value(word) {
            if (["digit", "teen", "tens"].includes(previous)) {
                return null;
            }
            current += number;
            saw_number = true;
            previous = "tens";
        } else if (word == "hundred" && saw_number && previous == "digit") {
            current *= 100;
            previous = "hundred";
        } else {
            return null;
        }
        index += 1;
    }
    if (saw_number) {
        current
    } else {
        null
    }
}

function spans_are_connected(text: string, left: WordSpan, right: WordSpan) -> bool {
    let separator = text.substring(left.end, right.start);
    separator.chars().every((character) -> {
        character.is_whitespace() || character == "-"
    })
}

function decimal_match(text: string, spans: WordSpan[], start: int) -> NumericMatch? {
    let point = start + 1;
    while (point < spans.length() && point <= start + 5) {
        if (!spans_are_connected(text, spans[point - 1], spans[point])) {
            return null;
        }
        if (spans[point].text.to_lower_case() == "point") {
            let integer_words = spans.slice(start, point).map((span) -> {
                span.text
            });
            if let integer: int = parse_spoken_number(integer_words) {
                let fraction = "";
                let end = point + 1;
                while (end < spans.length() && spans_are_connected(text, spans[end - 1], spans[end])) {
                    if let digit: int = digit_value(spans[end].text) {
                        fraction += `${digit}`;
                        end += 1;
                    } else {
                        break;
                    }
                }
                if (fraction != "") {
                    return NumericMatch { end_word: end, replacement: `${integer}.${fraction}` };
                }
            }
            return null;
        }
        point += 1;
    }
    null
}

function numeric_marker_match(text: string, spans: WordSpan[], start: int) -> NumericMatch? {
    if (spans[start].text.to_lower_case() != "numeric" || start + 1 >= spans.length()) {
        return null;
    }
    let end = start + 1;
    while (
        end < spans.length()
            && end <= start + 6
            && spans_are_connected(text, spans[end - 1], spans[end])
    ) {
        end += 1;
    }
    while (end > start + 1) {
        let words = spans.slice(start + 1, end).map((span) -> {
            span.text
        });
        if let number: int = parse_spoken_number(words) {
            return NumericMatch { end_word: end, replacement: `${number}` };
        }
        end -= 1;
    }
    null
}

function normalize_spoken_numerics(text: string) -> string {
    let spans = word_spans(text);
    let output = "";
    let cursor = 0;
    let word = 0;
    while (word < spans.length()) {
        let matched = numeric_marker_match(text, spans, word) ?? decimal_match(text, spans, word);
        if let replacement: NumericMatch = matched {
            output += text.substring(cursor, spans[word].start);
            output += replacement.replacement;
            cursor = spans[replacement.end_word - 1].end;
            word = replacement.end_word;
        } else {
            word += 1;
        }
    }
    output + text.substring(cursor, text.length())
}

function correction_matches(text: string, index: int, source: string) -> bool {
    let end = index + source.length();
    if (end > text.length() || text.substring(index, end).to_lower_case() != source.to_lower_case()) {
        return false;
    }
    let before_ok = index == 0 || !is_boundary_character(text.char_at(index - 1));
    let after_ok = end == text.length() || !is_boundary_character(text.char_at(end));
    before_ok && after_ok
}

function apply_guaranteed_corrections(text: string, rules: GlossaryRule[]) -> string {
    let ordered = rules
        .sort_by_key((rule) -> {
            rule.source.length()
        })
        .reverse();
    let output = "";
    let index = 0;
    while (index < text.length()) {
        let rule = ordered.find((candidate) -> {
            correction_matches(text, index, candidate.source)
        });
        if let matched: GlossaryRule = rule {
            output += matched.replacement;
            index += matched.source.length();
        } else {
            output += text.char_at(index);
            index += 1;
        }
    }
    output
}

function sentence_end_count(text: string) -> int {
    let count = 0;
    let index = 0;
    while (index < text.length()) {
        let character = text.char_at(index);
        if (character == "!" || character == "?") {
            count += 1;
        } else if (character == ".") {
            let previous_digit = index > 0 && text.char_at(index - 1).is_ascii_numeric();
            let next_digit = index + 1 < text.length() && text.char_at(index + 1).is_ascii_numeric();
            if (!(previous_digit && next_digit)) {
                count += 1;
            }
        }
        index += 1;
    }
    count
}

function leading_whitespace_length(text: string) -> int {
    let index = 0;
    while (index < text.length() && text.char_at(index).is_whitespace()) {
        index += 1;
    }
    index
}

function trailing_whitespace_start(text: string) -> int {
    let index = text.length();
    while (index > 0 && text.char_at(index - 1).is_whitespace()) {
        index -= 1;
    }
    index
}

function initial_word_end(text: string, start: int) -> int {
    let index = start;
    while (index < text.length() && text.char_at(index).is_ascii_alphabetic()) {
        index += 1;
    }
    index
}

function starts_with_personal_i(text: string, start: int, end: int) -> bool {
    if (text.substring(start, end).to_lower_case() != "i") {
        return false;
    }
    let rest = text.substring(end, text.length()).to_lower_case();
    if (rest == "") {
        return true;
    }
    if (
        ["'m", "'ve", "'ll", "'d"].some((prefix) -> {
            rest.starts_with(prefix)
        })
    ) {
        return true;
    }
    let next = rest.trim_start();
    if (next.length() == rest.length()) {
        return false;
    }
    let verbs = [
        "mean",
        "think",
        "guess",
        "believe",
        "know",
        "want",
        "need",
        "will",
        "would",
        "can",
        "could",
        "should",
        "am",
        "was",
        "have",
        "had",
        "do",
        "did",
        "feel",
        "see",
        "understand",
        "don't",
        "dont",
        "can't",
        "cant",
        "won't",
        "wont",
        "wouldn't",
        "wouldnt",
        "shouldn't",
        "shouldnt",
    ];
    verbs.some((verb) -> {
        next == verb
            || (next.starts_with(verb) && !is_boundary_character(next.char_at(verb.length())))
    })
}

function capitalize_initial_word(text: string, long_statement: bool) -> string {
    let start = leading_whitespace_length(text);
    let end = initial_word_end(text, start);
    if (end == start) {
        return text;
    }
    let word = text.substring(start, end);
    let replacement = if (starts_with_personal_i(text, start, end)) {
        "I"
    } else if (!long_statement && word == "A") {
        "a"
    } else if (
        !long_statement
            && word.char_at(0).is_ascii_uppercase()
            && word.substring(1, word.length()).is_ascii_lowercase()
    ) {
        word.to_lower_case()
    } else if (long_statement && word.is_ascii_lowercase()) {
        word.char_at(0).to_upper_case() + word.substring(1, word.length())
    } else {
        word
    };
    text.substring(0, start) + replacement + text.substring(end, text.length())
}

function normalize_short_statement_style(text: string) -> string {
    if (
        text.chars().some((character) -> {
            character.is_alphabetic() && !character.is_ascii_alphabetic()
        })
            || text.includes("?")
            || sentence_end_count(text) >= 2
    ) {
        return text;
    }
    let suffix_start = trailing_whitespace_start(text);
    let body = text.substring(0, suffix_start);
    let suffix = text.substring(suffix_start, text.length());
    if (word_count(text) > 10) {
        let styled = capitalize_initial_word(body, true);
        let punctuation = if (styled == "" || styled.ends_with(".") || styled.ends_with("!") || styled.ends_with("?")) {
            ""
        } else {
            "."
        };
        return styled + punctuation + suffix;
    }
    let without_period = if (body.ends_with(".")) {
        body.substring(0, body.length() - 1).trim_end()
    } else {
        body
    };
    capitalize_initial_word(without_period, false) + suffix
}

function normalize_final_transcript(text: string) -> string {
    normalize_short_statement_style(normalize_spoken_numerics(text))
}

function script_counts(text: string) -> int[] {
    let latin = 0;
    let non_latin = 0;
    for (let character in text.chars()) {
        if (character.is_alphabetic()) {
            if (character.is_ascii_alphabetic()) {
                latin += 1
            } else {
                non_latin += 1
            }
        }
    }
    [latin, non_latin]
}

function looks_like_unwanted_non_latin_translation(source: string, processed: string) -> bool {
    let source_counts = script_counts(source);
    let processed_counts = script_counts(processed);
    let allowed_growth = 6.max(source_counts[1] * 2);
    source_counts[0] > 0
        && processed_counts[1] > 0
        && processed_counts[1] > processed_counts[0]
        && processed_counts[1] > source_counts[1] + allowed_growth
}

function model_clean_attempt(transcript: string, glossary: string) -> ModelCleanAttempt {
    ModelCleanAttempt { result: clean_transcript(transcript, glossary), timed_out: false }
}

function timeout_attempt(timeout_ms: int) -> ModelCleanAttempt {
    baml.sys.sleep(baml.time.Duration.from_milliseconds(timeout_ms)) catch_all (error) {
        _ => null,
    };
    ModelCleanAttempt { result: null, timed_out: true }
}

function clean_with_timeout(
    transcript: string,
    glossary: string,
    timeout_seconds: float,
) -> ModelCleanAttempt {
    let timeout_ms = (timeout_seconds * 1000.0).iceil();
    let model = spawn { model_clean_attempt(transcript, glossary) };
    let timeout = spawn { timeout_attempt(timeout_ms) };
    await baml.future.race([model, timeout])
}

function process_transcript(
    text: string,
    model_enabled: bool,
    timeout_seconds: float,
    glossary_file: string?,
) -> string {
    let raw_word_count = word_count(text);
    let glossary = load_glossary(glossary_file) catch_all (error) {
        _ => {
            baml.io.eprintln(`Warning: ${error.to_string()}; using local cleanup without glossary.`);
            empty_glossary()
        },
    };
    let prepared = apply_guaranteed_corrections(normalize_spoken_numerics(text), glossary.always);
    let local = normalize_short_statement_style(prepared);
    if (!should_clean_with_model(raw_word_count, model_enabled)) {
        return local;
    }

    let attempt = clean_with_timeout(prepared, glossary_prompt(glossary), timeout_seconds);
    if (attempt.timed_out) {
        baml.io.eprintln(
            `Warning: transcript post-processing timed out after ${timeout_seconds}s; using local cleanup.`,
        );
        return local;
    }
    let result = attempt.result ?? CleanResult { text: null, error: "missing model result" };
    let cleaned = result.text ?? "";
    if (cleaned.trim() == "") {
        baml.io.eprintln(
            `Warning: transcript post-processing failed: ${result.error ?? "empty model output"}; using local cleanup.`,
        );
        return local;
    }
    if (looks_like_unwanted_non_latin_translation(prepared, cleaned)) {
        baml.io.eprintln("Warning: transcript cleanup changed the language; using local cleanup.");
        return local;
    }
    normalize_final_transcript(apply_guaranteed_corrections(cleaned, glossary.always))
}

test "BAML normalizes spoken numbers" {
    assert.equal(normalize_spoken_numerics("zero point one"), "0.1");
    assert.equal(normalize_spoken_numerics("version twelve point zero"), "version 12.0");
    assert.equal(normalize_spoken_numerics("one hundred and five point six"), "105.6");
    assert.equal(normalize_spoken_numerics("numeric twenty one"), "21");
    assert.equal(normalize_spoken_numerics("one and two point three"), "one and 2.3")
}

test "BAML guaranteed rules are boundary aware and do not cascade" {
    let rules = [
        GlossaryRule { source: "code", replacement: "Codex" },
        GlossaryRule { source: "cloud code", replacement: "Claude Code" },
        GlossaryRule { source: "cat", replacement: "dog" },
    ];
    assert.equal(
        apply_guaranteed_corrections("Cloud code and cat scatter", rules),
        "Claude Code and dog scatter",
    )
}

test "BAML preserves established statement style" {
    assert.equal(normalize_final_transcript("Fair point."), "fair point");
    assert.equal(
        normalize_final_transcript("Because it will be simpler this way."),
        "because it will be simpler this way",
    );
    assert.equal(normalize_final_transcript("Version zero point one."), "version 0.1");
    assert.equal(normalize_final_transcript("A fair point."), "a fair point");
    assert.equal(normalize_final_transcript("i mean"), "I mean");
    assert.equal(normalize_final_transcript("i'm sure"), "I'm sure");
    assert.equal(normalize_final_transcript("It's fine."), "it's fine");
    assert.equal(normalize_final_transcript("API request."), "API request");
    assert.equal(normalize_final_transcript("Use API."), "use API");
    assert.equal(normalize_final_transcript("for i in items"), "for i in items");
    assert.equal(normalize_final_transcript("TypeScript type."), "TypeScript type");
    assert.equal(normalize_final_transcript("How can we solve it?"), "How can we solve it?");
    assert.equal(
        normalize_final_transcript("That's a fair point. Let's go with this approach."),
        "That's a fair point. Let's go with this approach.",
    );
    assert.equal(
        normalize_final_transcript("Хорошая мысль."),
        "Хорошая мысль.",
    );
    assert.equal(
        normalize_final_transcript("because it will be simpler this way and it reduces complexity overall"),
        "Because it will be simpler this way and it reduces complexity overall.",
    );
    assert.equal(
        normalize_final_transcript("i think this approach will be simpler because it reduces complexity overall"),
        "I think this approach will be simpler because it reduces complexity overall.",
    );
    assert.equal(
        normalize_final_transcript("TypeScript type inference should stay unchanged when it starts the statement"),
        "TypeScript type inference should stay unchanged when it starts the statement.",
    )
}

test "BAML parses the system glossary shape" {
    let glossary = parse_glossary(
        "[always]\nengine x -> nginx\n[likely]\ncloud code -> Claude Code\n[contextual]\ncodecs -> Codex\n[terms]\nTypeScript\n",
    );
    assert.equal(glossary.always[0], GlossaryRule { source: "engine x", replacement: "nginx" });
    assert.equal(glossary_prompt(glossary).includes("<canonical_terms>\nTypeScript"), true)
}

test "BAML rejects duplicate glossary sources" {
    let failed = parse_glossary("[always]\ncodecs -> Codex\n[contextual]\ncodecs -> Codex") catch_all (error) {
        _ => true
    };
    assert.equal(failed, true)
}
