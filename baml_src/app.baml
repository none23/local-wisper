enum AppCommand {
    Record,
    Preload,
    SwayStart,
    SwayStop,
    SwayCancel,
}

enum DevicePreference {
    Auto,
    Cuda,
    Cpu,
}

enum DeliveryMode {
    Copy,
    Type,
}

class AppOptions {
    command: AppCommand,
    device: DevicePreference,
    type_output: bool,
    post_process_model: bool,
    post_process_timeout: float,
    post_process_glossary_file: string?,
}

function invalid_argument(message: string) -> never {
    throw baml.errors.InvalidArgument { message: message }
}

function required_value(args: string[], index: int, flag: string) -> string {
    args.at(index) ?? invalid_argument(`${flag} requires a value`)
}

function parse_device(value: string) -> DevicePreference {
    match (value) {
        "auto" => DevicePreference.Auto,
        "cuda" => DevicePreference.Cuda,
        "cpu" => DevicePreference.Cpu,
        _ => invalid_argument(`invalid --device value: ${value}`),
    }
}

function parse_command(value: string) -> AppCommand? {
    match (value) {
        "record" => AppCommand.Record,
        "preload" => AppCommand.Preload,
        "sway-start" => AppCommand.SwayStart,
        "sway-stop" => AppCommand.SwayStop,
        "sway-cancel" => AppCommand.SwayCancel,
        _ => null,
    }
}

function parse_options(args: string[]) -> AppOptions {
    let options = AppOptions {
        command: AppCommand.Record,
        device: DevicePreference.Auto,
        type_output: false,
        post_process_model: false,
        post_process_timeout: 20.0,
        post_process_glossary_file: null,
    };
    let command_seen = false;
    let index = 0;
    while (index < args.length()) {
        let arg = args[index];
        match (arg) {
            "--backend" => {
                let value = required_value(args, index + 1, arg);
                if (value != "parakeet") {
                    invalid_argument("only --backend parakeet is supported")
                }
                index += 2
            },
            "--model" => {
                let value = required_value(args, index + 1, arg);
                if (value != "nvidia/parakeet-tdt-0.6b-v3") {
                    invalid_argument("only --model nvidia/parakeet-tdt-0.6b-v3 is supported")
                }
                index += 2
            },
            "--compute-type" => {
                let _ = required_value(args, index + 1, arg);
                index += 2
            },
            "--device" => {
                options.device = parse_device(required_value(args, index + 1, arg));
                index += 2
            },
            "--sample-rate" => {
                let value = required_value(args, index + 1, arg);
                if (value != "16000") {
                    invalid_argument("only --sample-rate 16000 is supported")
                }
                index += 2
            },
            "--vad-filter" => invalid_argument("VAD is not supported; use --no-vad-filter"),
            "--no-vad-filter" => {
                index += 1
            },
            "--type-output" => {
                options.type_output = true;
                index += 1
            },
            "--post-process-model" => {
                let value = required_value(args, index + 1, arg);
                if (value != "gpt-5.6-luna") {
                    invalid_argument("only --post-process-model gpt-5.6-luna is supported")
                }
                options.post_process_model = true;
                index += 2
            },
            "--post-process-timeout" => {
                let value = baml.Float.parse(required_value(args, index + 1, arg));
                if (!value.is_finite() || value <= 0.0) {
                    invalid_argument("--post-process-timeout must be greater than zero")
                }
                options.post_process_timeout = value;
                index += 2
            },
            "--post-process-glossary-file" => {
                options.post_process_glossary_file = required_value(args, index + 1, arg);
                index += 2
            },
            "--help" | "-h" => {
                baml.io.println("Usage: lw [record|preload|sway-start|sway-stop|sway-cancel]");
                baml.sys.exit(0)
            },
            _ => {
                let command = parse_command(arg);
                if let parsed: AppCommand = command {
                    if (command_seen) {
                        invalid_argument(`unexpected second command: ${arg}`)
                    }
                    options.command = parsed;
                    command_seen = true;
                    index += 1
                } else {
                    invalid_argument(`unknown argument: ${arg}`)
                }
            },
        }
    }
    options
}

function clean_if_present(
    transcript: string,
    options: AppOptions,
    native_clean: (
        transcript: string,
        model_enabled: bool,
        timeout_seconds: float,
        glossary_file: string?,
    ) -> string throws baml.errors.HostCallable,
) -> string {
    if (transcript.trim() == "") {
        baml.io.eprintln("No speech detected.");
        ""
    } else {
        native_clean(
            transcript,
            options.post_process_model,
            options.post_process_timeout,
            options.post_process_glossary_file,
        )
    }
}

function run_workflow(
    options: AppOptions,
    native_ensure_model: (device: DevicePreference) -> null throws baml.errors.HostCallable,
    native_start_model: (device: DevicePreference) -> null throws baml.errors.HostCallable,
    native_record_interactively: () -> string throws baml.errors.HostCallable,
    native_sway_start: () -> null throws baml.errors.HostCallable,
    native_sway_stop: () -> string throws baml.errors.HostCallable,
    native_sway_cancel: () -> null throws baml.errors.HostCallable,
    native_transcribe: (
        audio_path: string,
        device: DevicePreference,
    ) -> string throws baml.errors.HostCallable,
    native_clean: (
        transcript: string,
        model_enabled: bool,
        timeout_seconds: float,
        glossary_file: string?,
    ) -> string throws baml.errors.HostCallable,
    native_deliver: (transcript: string, mode: DeliveryMode) -> bool throws baml.errors.HostCallable,
) -> null {
    match (options.command) {
        AppCommand.Record => {
            //# Load once, then let BAML own the record-to-delivery sequence.
            native_ensure_model(options.device);
            let audio = native_record_interactively();
            let transcript = clean_if_present(native_transcribe(audio, options.device), options, native_clean);
            if (transcript != "") {
                baml.io.println(transcript);
                if (!native_deliver(transcript, DeliveryMode.Copy)) {
                    baml.io.eprintln("Warning: Could not copy transcript to the clipboard.")
                }
            }
            null
        },
        AppCommand.Preload => native_ensure_model(options.device),
        AppCommand.SwayStart => {
            native_sway_start();
            native_start_model(options.device)
        },
        AppCommand.SwayStop => {
            let audio = native_sway_stop();
            let transcript = clean_if_present(native_transcribe(audio, options.device), options, native_clean);
            if (transcript != "") {
                baml.io.println(transcript);
                let mode = if (options.type_output) {
                    DeliveryMode.Type
                } else {
                    DeliveryMode.Copy
                };
                if (!native_deliver(transcript, mode)) {
                    baml.io.eprintln(
                        "Warning: Could not deliver transcript to the focused application.",
                    )
                }
            }
            null
        },
        AppCommand.SwayCancel => native_sway_cancel(),
    }
}

function run_app(
    args: string[],
    native_ensure_model: (device: DevicePreference) -> null throws baml.errors.HostCallable,
    native_start_model: (device: DevicePreference) -> null throws baml.errors.HostCallable,
    native_record_interactively: () -> string throws baml.errors.HostCallable,
    native_sway_start: () -> null throws baml.errors.HostCallable,
    native_sway_stop: () -> string throws baml.errors.HostCallable,
    native_sway_cancel: () -> null throws baml.errors.HostCallable,
    native_transcribe: (
        audio_path: string,
        device: DevicePreference,
    ) -> string throws baml.errors.HostCallable,
    native_clean: (
        transcript: string,
        model_enabled: bool,
        timeout_seconds: float,
        glossary_file: string?,
    ) -> string throws baml.errors.HostCallable,
    native_deliver: (transcript: string, mode: DeliveryMode) -> bool throws baml.errors.HostCallable,
) -> int {
    run_workflow(
        parse_options(args),
        native_ensure_model,
        native_start_model,
        native_record_interactively,
        native_sway_start,
        native_sway_stop,
        native_sway_cancel,
        native_transcribe,
        native_clean,
        native_deliver,
    ) catch_all (error) {
        _ => {
            baml.io.eprintln(error.to_string());
            return 1;
        },
    };
    0
}

test "BAML parses the unchanged Sway invocation" {
    let options = parse_options(
        [
            "--backend",
            "parakeet",
            "--device",
            "cuda",
            "--sample-rate",
            "16000",
            "--compute-type",
            "float16",
            "--no-vad-filter",
            "--type-output",
            "sway-stop",
        ],
    );
    assert.equal(options.command, AppCommand.SwayStop);
    assert.equal(options.device, DevicePreference.Cuda);
    assert.equal(options.type_output, true)
}

test "bare arguments select interactive recording" {
    assert.equal(parse_options([]).command, AppCommand.Record)
}
