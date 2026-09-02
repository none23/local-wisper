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

class ModelCleanAttempt {
    result: CleanResult?,
    timed_out: bool,
}

enum TranscriptProcessing {
    Local,
    Model,
    ModelTimeoutFallback,
    ModelErrorFallback,
    ModelRejectedFallback,
}

class ProcessedTranscript {
    text: string,
    processing: TranscriptProcessing,
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
            current = glossary_section(line.slice(1, line.length() - 1).trim(), line_number);
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
        let source = line.slice(0, arrow).trim();
        let replacement = line.slice(arrow + 2, line.length()).trim();
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
            null => {
                invalid_argument(`glossary entry on line ${line_number} must follow a section header`)
            },
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

function model_clean_attempt(
    transcript: string,
    glossary: string,
    model: string,
) -> ModelCleanAttempt {
    ModelCleanAttempt { result: clean_transcript(transcript, glossary, model), timed_out: false }
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
    model_name: string,
    timeout_seconds: float,
) -> ModelCleanAttempt {
    let timeout_ms = (timeout_seconds * 1000.0).iceil();
    let model = spawn { model_clean_attempt(transcript, glossary, model_name) };
    let timeout = spawn { timeout_attempt(timeout_ms) };
    await baml.future.race([model, timeout])
}

function process_transcript(
    text: string,
    model: string?,
    timeout_seconds: float,
    glossary_file: string?,
) -> ProcessedTranscript {
    let original = text.trim();
    let model_name = model
        ?? return ProcessedTranscript { text: original, processing: TranscriptProcessing.Local };
    let glossary = load_glossary(glossary_file) catch_all (error) {
        _ => {
            baml.io.eprintln(`Warning: ${error.to_string()}; cleaning without glossary.`);
            empty_glossary()
        },
    };
    let attempt = clean_with_timeout(text, glossary_prompt(glossary), model_name, timeout_seconds);
    if (attempt.timed_out) {
        baml.io.eprintln(
            `Warning: transcript post-processing timed out after ${timeout_seconds}s; using original transcript.`,
        );
        return ProcessedTranscript {
            text: original,
            processing: TranscriptProcessing.ModelTimeoutFallback,
        };
    }
    let result = attempt.result ?? CleanResult { text: null, error: "missing model result" };
    let cleaned = result.text ?? "";
    if (cleaned.trim() == "") {
        baml.io.eprintln(
            `Warning: transcript post-processing failed: ${result.error ?? "empty model output"}; using original transcript.`,
        );
        return ProcessedTranscript { text: original, processing: TranscriptProcessing.ModelErrorFallback };
    }
    if (looks_like_unwanted_non_latin_translation(text, cleaned)) {
        baml.io.eprintln(
            "Warning: transcript cleanup changed the language; using original transcript.",
        );
        return ProcessedTranscript {
            text: original,
            processing: TranscriptProcessing.ModelRejectedFallback,
        };
    }
    ProcessedTranscript { text: cleaned.trim(), processing: TranscriptProcessing.Model }
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
