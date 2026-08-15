class DaemonState {
    address: string,
    token: string,
}

class TranscribeRequest {
    audio_path: string,
}

class DaemonResponse {
    ok: bool,
    text: string?,
    error: string?,
}

function daemon_state_path(runtime_dir: string) -> string {
    join_path(runtime_dir, "daemon.json")
}

function daemon_error_path(runtime_dir: string) -> string {
    join_path(runtime_dir, "daemon.error")
}

function spawn_model_daemon(
    preference: DevicePreference,
    native_spawn_daemon: (
        preference: DevicePreference,
        log_path: string,
    ) -> null throws baml.errors.HostCallable,
) -> null {
    native_spawn_daemon(preference, join_path(cache_root(), "daemon.log"))
}

function write_daemon_state(runtime_dir: string, state: DaemonState) -> null {
    atomic_write(daemon_state_path(runtime_dir), baml.json.to_string(state))
}

function read_daemon_state(runtime_dir: string) -> DaemonState? {
    let path = daemon_state_path(runtime_dir);
    if (!baml.fs.exists(path)) {
        return null;
    }
    baml.json.from_string<DaemonState>(baml.fs.read(path)) catch_all (error) {
        _ => null
    }
}

function json_response(status: int, response: DaemonResponse) -> baml.http.Response {
    baml.http.Response.new(
        status,
        { "content-type": "application/json" },
        baml.json.to_string(response).to_utf8(),
    )
}

function handle_daemon_request(
    request: baml.http.Request,
    token: string,
    native_transcribe_loaded: (audio_path: string) -> string throws baml.errors.HostCallable,
) -> baml.http.Response {
    if (request.headers.get("x-local-wisper-token") != token) {
        return json_response(403, DaemonResponse { ok: false, text: null, error: "forbidden" });
    }
    if (request.method == "GET" && request.url == "/ping") {
        return json_response(200, DaemonResponse { ok: true, text: null, error: null });
    }
    if (request.method != "POST" || request.url != "/transcribe") {
        return json_response(404, DaemonResponse { ok: false, text: null, error: "not found" });
    }
    let response = baml.json.from_string<TranscribeRequest>(request.body) catch_all (error) {
        _ => {
            return json_response(400, DaemonResponse { ok: false, text: null, error: error.to_string() });
        },
    };
    let transcript = native_transcribe_loaded(response.audio_path) catch_all (error) {
        _ => {
            return json_response(500, DaemonResponse { ok: false, text: null, error: error.to_string() });
        },
    };
    json_response(200, DaemonResponse { ok: true, text: transcript, error: null })
}

function serve_daemon(
    runtime_dir: string,
    preference: DevicePreference,
    native_acquire_model_lock: () -> bool throws baml.errors.HostCallable,
    native_load_model: (
        model_dir: string,
        variant: ModelVariant,
    ) -> null throws baml.errors.HostCallable,
    native_transcribe_loaded: (audio_path: string) -> string throws baml.errors.HostCallable,
) -> null {
    if (!native_acquire_model_lock()) {
        return null;
    }
    remove_file_if_present(daemon_error_path(runtime_dir));
    initialize_model(preference, native_load_model);
    let server = baml.http.Server.bind("127.0.0.1:0");
    let state = DaemonState { address: server.addr, token: baml.id.new() };
    write_daemon_state(runtime_dir, state);
    baml.io.eprintln(`ready on ${state.address}`);
    server.serve((request) -> {
        handle_daemon_request(request, state.token, native_transcribe_loaded) catch_all (error) {
            _ => {
                baml.http.Response.new(
                    500,
                    { "content-type": "text/plain" },
                    "internal daemon error".to_utf8(),
                )
            },
        }
    })
}

function run_daemon(
    runtime_dir: string,
    preference: DevicePreference,
    native_acquire_model_lock: () -> bool throws baml.errors.HostCallable,
    native_load_model: (
        model_dir: string,
        variant: ModelVariant,
    ) -> null throws baml.errors.HostCallable,
    native_transcribe_loaded: (audio_path: string) -> string throws baml.errors.HostCallable,
) -> int {
    serve_daemon(
        runtime_dir,
        preference,
        native_acquire_model_lock,
        native_load_model,
        native_transcribe_loaded,
    ) catch_all (error) {
        _ => {
            let _ = baml.fs.write(daemon_error_path(runtime_dir), `${error.to_string()}\n`);
            baml.io.eprintln(error.to_string());
            return 1;
        },
    };
    0
}

function daemon_request(
    state: DaemonState,
    method: string,
    path: string,
    body: string,
    timeout_ms: int,
) -> DaemonResponse {
    let response = baml.http.send(
        baml.http.Request {
            method: method,
            url: `http://${state.address}${path}`,
            headers: { "content-type": "application/json", "x-local-wisper-token": state.token },
            body: body,
        },
        timeout = baml.time.Duration.from_milliseconds(timeout_ms),
    );
    baml.json.from_string<DaemonResponse>(response.text())
}

function ping_daemon(runtime_dir: string) -> bool {
    if let state: DaemonState = read_daemon_state(runtime_dir) {
        let response = daemon_request(state, "GET", "/ping", "", 250) catch_all (error) {
            _ => {
                return false;
            },
        };
        response.ok
    } else {
        false
    }
}

function start_model_daemon(
    runtime_dir: string,
    preference: DevicePreference,
    native_spawn_daemon: (
        preference: DevicePreference,
        log_path: string,
    ) -> null throws baml.errors.HostCallable,
) -> null {
    if (!ping_daemon(runtime_dir)) {
        spawn_model_daemon(preference, native_spawn_daemon)
    }
    null
}

function ensure_model_daemon(
    runtime_dir: string,
    preference: DevicePreference,
    native_spawn_daemon: (
        preference: DevicePreference,
        log_path: string,
    ) -> null throws baml.errors.HostCallable,
) -> null {
    if (ping_daemon(runtime_dir)) {
        return null;
    }
    let error_path = daemon_error_path(runtime_dir);
    remove_file_if_present(error_path);
    spawn_model_daemon(preference, native_spawn_daemon);
    let started = baml.time.Instant.now();
    let last_spawn = baml.time.Instant.now();
    while (started.elapsed().to_milliseconds() < 300000n) {
        if (ping_daemon(runtime_dir)) {
            return null;
        }
        if (baml.fs.exists(error_path)) {
            invalid_argument(`transcription daemon failed to start: ${baml.fs.read(error_path).trim()}`)
        }
        if (last_spawn.elapsed().to_milliseconds() >= 3000n) {
            spawn_model_daemon(preference, native_spawn_daemon);
            last_spawn = baml.time.Instant.now();
        }
        baml.sys.sleep(baml.time.Duration.from_milliseconds(150));
    }
    invalid_argument("transcription daemon did not become ready within 300 seconds")
}

function transcribe_with_daemon(
    runtime_dir: string,
    audio_path: string,
    preference: DevicePreference,
    native_spawn_daemon: (
        preference: DevicePreference,
        log_path: string,
    ) -> null throws baml.errors.HostCallable,
) -> string {
    ensure_model_daemon(runtime_dir, preference, native_spawn_daemon);
    let state = read_daemon_state(runtime_dir) ?? invalid_argument("daemon state disappeared");
    let response = daemon_request(
        state,
        "POST",
        "/transcribe",
        baml.json.to_string(TranscribeRequest { audio_path: audio_path }),
        120000,
    );
    if (!response.ok) {
        invalid_argument(`transcription failed: ${response.error ?? "unknown error"}`)
    }
    response.text ?? ""
}

test "daemon state round trips through JSON" {
    let state = DaemonState { address: "127.0.0.1:1234", token: "secret" };
    assert.equal(baml.json.from_string<DaemonState>(baml.json.to_string(state)), state)
}

test "daemon handler authenticates before invoking native inference" {
    let request = baml.http.Request {
        method: "POST",
        url: "/transcribe",
        headers: { "x-local-wisper-token": "wrong" },
        body: baml.json.to_string(TranscribeRequest { audio_path: "/tmp/test.wav" }),
    };
    let response = handle_daemon_request(request, "secret", (audio_path) -> {
        `transcribed ${audio_path}`
    });
    assert.equal(response.status_code, 403)
}

test "daemon handler routes authenticated transcription" {
    let request = baml.http.Request {
        method: "POST",
        url: "/transcribe",
        headers: { "x-local-wisper-token": "secret" },
        body: baml.json.to_string(TranscribeRequest { audio_path: "/tmp/test.wav" }),
    };
    let response = handle_daemon_request(request, "secret", (audio_path) -> {
        `transcribed ${audio_path}`
    });
    assert.equal(response.status_code, 200)
}
