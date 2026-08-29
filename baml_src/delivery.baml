function process_options(stdin: string?) -> baml.sys.ProcessOptions {
    baml.sys.ProcessOptions {
        cwd: null,
        env: null,
        timeout_ms: 10000,
        stdin: stdin,
        keep_stdin_open: false,
    }
}

function run_delivery_command(program: string, args: string[], stdin: string?) -> bool {
    let output = baml.sys.exec(program, args, process_options(stdin)) catch_all (error) {
        _ => {
            return false;
        },
    };
    output.ok()
}

// Chromium treats wtype's fourteenth generated keycode as Backspace.
function wtype_chunks(transcript: string) -> string[] {
    let chunks: string[] = [];
    let chunk = "";
    let characters: string[] = [];
    for (let character in transcript.chars()) {
        if (!characters.includes(character) && characters.length() == 13) {
            chunks.push(chunk);
            chunk = "";
            characters = [];
        }
        chunk += character;
        if (!characters.includes(character)) {
            characters.push(character)
        }
    }
    if (chunk != "") {
        chunks.push(chunk)
    }
    chunks
}

function type_text(transcript: string) -> bool {
    for (let chunk in wtype_chunks(transcript)) {
        if (!run_delivery_command("wtype", ["-"], chunk)) {
            return false;
        }
        baml.sys.sleep(baml.time.Duration.from_milliseconds(5)) catch_all (error) {
            _ => {
                return false;
            },
        }
    }
    true
}

function deliver_text(transcript: string, mode: DeliveryMode) -> bool {
    match (mode) {
        DeliveryMode.Type => type_text(transcript),
        DeliveryMode.Copy => {
            run_delivery_command("wl-copy", [], transcript)
                || run_delivery_command("xclip", ["-selection", "clipboard"], transcript)
                || run_delivery_command("xsel", ["--clipboard", "--input"], transcript)
        },
    }
}

test "wtype chunks avoid the Chromium backspace keycode" {
    assert.equal(
        wtype_chunks("Okay, I think we're ready to do steps three and four."),
        ["Okay, I think we", "'re ready to do steps three and ", "four."],
    )
}

test "wtype chunks split only for a fourteenth distinct character" {
    assert.equal(wtype_chunks("abcdefghijklmmlkjihgfedcba"), ["abcdefghijklmmlkjihgfedcba"]);
    assert.equal(wtype_chunks("abcdefghijklmn"), ["abcdefghijklm", "n"])
}
