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
    post_process_model: string?,
    post_process_timeout: float,
    post_process_glossary_file: string?,
}

class TranscriptRecord {
    raw_text: string,
    final_text: string,
    processing: TranscriptProcessing,
    post_process_model: string?,
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
        post_process_model: null,
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
                if (value.trim() == "") {
                    invalid_argument("--post-process-model must not be empty")
                }
                options.post_process_model = value.trim();
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
                baml.io.println(
                    "Usage: lw [record|preload|sway-start|sway-stop|sway-cancel] [options]",
                );
                baml.io.println("  --post-process-model MODEL");
                baml.io.println("  --post-process-timeout SECONDS");
                baml.io.println("  --post-process-glossary-file PATH");
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

function prepare_transcript(raw_text: string, options: AppOptions) -> TranscriptRecord? {
    if (raw_text.trim() == "") {
        baml.io.eprintln("No speech detected.");
        null
    } else {
        let processed = process_transcript(
            raw_text,
            options.post_process_model,
            options.post_process_timeout,
            options.post_process_glossary_file,
        );
        TranscriptRecord {
            raw_text: raw_text,
            final_text: processed.text,
            processing: processed.processing,
            post_process_model: options.post_process_model,
        }
    }
}

function store_transcript(
    transcript: TranscriptRecord,
    native_store_transcript: (transcript: TranscriptRecord) -> null throws baml.errors.HostCallable,
) -> null {
    native_store_transcript(transcript) catch_all (error) {
        _ => {
            baml.io.eprintln(`Warning: Could not save transcript history: ${error.to_string()}`);
            null
        },
    }
}

function delivery_failure_message(mode: DeliveryMode) -> string {
    match (mode) {
        DeliveryMode.Copy => "Could not copy transcript to the clipboard.",
        DeliveryMode.Type => "Could not type transcript into the focused application.",
    }
}

function complete_transcription(
    raw_text: string,
    options: AppOptions,
    mode: DeliveryMode,
    native_store_transcript: (transcript: TranscriptRecord) -> null throws baml.errors.HostCallable,
) -> null {
    if let transcript: TranscriptRecord = prepare_transcript(raw_text, options) {
        baml.io.println(transcript.final_text);
        if (!deliver_text(transcript.final_text, mode)) {
            baml.io.eprintln(`Warning: ${delivery_failure_message(mode)}`)
        }
        //# Persistence is best effort and must not delay delivery.
        store_transcript(transcript, native_store_transcript)
    } else {
        null
    }
}

function run_workflow(
    options: AppOptions,
    native_spawn_daemon: (
        preference: DevicePreference,
        log_path: string,
    ) -> null throws baml.errors.HostCallable,
    native_runtime_dir: () -> string throws baml.errors.HostCallable,
    native_spawn_recorder: (
        backend: RecorderBackend,
        audio_path: string,
        log_path: string,
    ) -> NativeRecorder throws baml.errors.HostCallable,
    native_recorder_exists: (process: NativeRecorder) -> bool throws baml.errors.HostCallable,
    native_stop_recorder: (
        process: NativeRecorder,
        backend: RecorderBackend,
    ) -> null throws baml.errors.HostCallable,
    native_store_transcript: (transcript: TranscriptRecord) -> null throws baml.errors.HostCallable,
) -> null {
    match (options.command) {
        AppCommand.Record => {
            //# Load once, then let BAML own the record-to-delivery sequence.
            let runtime_dir = native_runtime_dir();
            ensure_model_daemon(runtime_dir, options.device, native_spawn_daemon);
            let audio = record_interactively(runtime_dir, native_spawn_recorder, native_recorder_exists, native_stop_recorder);
            defer { cleanup_audio(audio) }
            complete_transcription(
                transcribe_with_daemon(runtime_dir, audio.path, options.device, native_spawn_daemon),
                options,
                DeliveryMode.Copy,
                native_store_transcript,
            );
            null
        },
        AppCommand.Preload => {
            ensure_model_daemon(native_runtime_dir(), options.device, native_spawn_daemon)
        },
        AppCommand.SwayStart => {
            let runtime_dir = native_runtime_dir();
            sway_start_recording(runtime_dir, native_spawn_recorder, native_recorder_exists, native_stop_recorder);
            start_model_daemon(runtime_dir, options.device, native_spawn_daemon)
        },
        AppCommand.SwayStop => {
            let runtime_dir = native_runtime_dir();
            let audio = sway_stop_recording(runtime_dir, native_recorder_exists, native_stop_recorder);
            defer { cleanup_audio(audio) }
            let mode = if (options.type_output) {
                DeliveryMode.Type
            } else {
                DeliveryMode.Copy
            };
            complete_transcription(
                transcribe_with_daemon(runtime_dir, audio.path, options.device, native_spawn_daemon),
                options,
                mode,
                native_store_transcript,
            );
            null
        },
        AppCommand.SwayCancel => {
            sway_cancel_recording(native_runtime_dir(), native_recorder_exists, native_stop_recorder)
        },
    }
}

function run_app(
    args: string[],
    native_spawn_daemon: (
        preference: DevicePreference,
        log_path: string,
    ) -> null throws baml.errors.HostCallable,
    native_runtime_dir: () -> string throws baml.errors.HostCallable,
    native_spawn_recorder: (
        backend: RecorderBackend,
        audio_path: string,
        log_path: string,
    ) -> NativeRecorder throws baml.errors.HostCallable,
    native_recorder_exists: (process: NativeRecorder) -> bool throws baml.errors.HostCallable,
    native_stop_recorder: (
        process: NativeRecorder,
        backend: RecorderBackend,
    ) -> null throws baml.errors.HostCallable,
    native_store_transcript: (transcript: TranscriptRecord) -> null throws baml.errors.HostCallable,
) -> int {
    run_workflow(
        parse_options(args),
        native_spawn_daemon,
        native_runtime_dir,
        native_spawn_recorder,
        native_recorder_exists,
        native_stop_recorder,
        native_store_transcript,
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

test "post-processing model is user configurable" {
    let options = parse_options(["--post-process-model", "gpt-5.6-luna-next"]);
    assert.equal(options.post_process_model, "gpt-5.6-luna-next")
}

test "history receives the raw and final transcript" {
    let saved: string[] = [];
    store_transcript(TranscriptRecord { raw_text: "raw words", final_text: "Final words.", processing: TranscriptProcessing.Model, post_process_model: "gpt-5.6-luna" }, (
        transcript,
    ) -> {
        saved.push(
            `${transcript.raw_text}|${transcript.final_text}|${transcript.processing}|${transcript.post_process_model ?? "none"}`,
        );
        null
    });
    assert.equal(saved, ["raw words|Final words.|Model|gpt-5.6-luna"])
}

test "local processing records the raw and final transcript" {
    let transcript = prepare_transcript("version zero point one.", parse_options([]));
    assert.equal(
        transcript,
        TranscriptRecord {
            raw_text: "version zero point one.",
            final_text: "version 0.1",
            processing: TranscriptProcessing.Local,
            post_process_model: null,
        },
    )
}
