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

function deliver_text(transcript: string, mode: DeliveryMode) -> bool {
    match (mode) {
        DeliveryMode.Type => run_delivery_command("wtype", [transcript], null),
        DeliveryMode.Copy => {
            run_delivery_command("wl-copy", [], transcript)
                || run_delivery_command("xclip", ["-selection", "clipboard"], transcript)
                || run_delivery_command("xsel", ["--clipboard", "--input"], transcript)
        },
    }
}
