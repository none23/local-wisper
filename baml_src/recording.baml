enum RecorderBackend {
    PwRecord,
    Ffmpeg,
}

class NativeRecorder {
    pid: int,
    started_at: int,
}

class RecordingState {
    process: NativeRecorder,
    backend: RecorderBackend,
    audio_path: string,
    session_dir: string,
}

class RecordedAudio {
    path: string,
    session_dir: string,
}

function recording_state_path(runtime_dir: string) -> string {
    join_path(runtime_dir, "recording.json")
}

function cleanup_recording_files(state_path: string, state: RecordingState) -> null {
    remove_file_if_present(state_path);
    baml.fs.remove_dir_all(state.session_dir) catch_all (error) {
        _ => null
    }
}

function cleanup_audio(audio: RecordedAudio) -> null {
    baml.fs.remove_dir_all(audio.session_dir) catch_all (error) {
        _ => null
    }
}

function read_recording_state(path: string) -> RecordingState? {
    if (!baml.fs.exists(path)) {
        return null;
    }
    baml.json.from_string<RecordingState>(baml.fs.read(path))
}

function commit_recording_state(path: string, state: RecordingState) -> null {
    atomic_write(path, baml.json.to_string(state))
}

function validate_recorded_audio(audio_path: string) -> null {
    if (!baml.fs.exists(audio_path) || baml.fs.size(audio_path) < 2048) {
        invalid_argument("Recording is empty or too short to transcribe")
    }
    null
}

function finish_recording(state: RecordingState) -> RecordedAudio {
    validate_recorded_audio(state.audio_path) catch_all (error) {
        _ => {
            baml.fs.remove_dir_all(state.session_dir) catch_all (cleanup_error) {
                _ => null
            };
            throw error;
        },
    };
    RecordedAudio { path: state.audio_path, session_dir: state.session_dir }
}

function try_start_recorder(
    backend: RecorderBackend,
    audio_path: string,
    log_path: string,
    session_dir: string,
    native_spawn_recorder: (
        backend: RecorderBackend,
        audio_path: string,
        log_path: string,
    ) -> NativeRecorder throws baml.errors.HostCallable,
    native_recorder_exists: (process: NativeRecorder) -> bool throws baml.errors.HostCallable,
) -> RecordingState? {
    let process = native_spawn_recorder(backend, audio_path, log_path) catch_all (error) {
        _ => {
            return null;
        },
    };
    let delay = match (backend) {
        RecorderBackend.PwRecord => 250,
        RecorderBackend.Ffmpeg => 400,
    };
    baml.sys.sleep(baml.time.Duration.from_milliseconds(delay));
    if (!native_recorder_exists(process)) {
        return null;
    }
    RecordingState {
        process: process,
        backend: backend,
        audio_path: audio_path,
        session_dir: session_dir,
    }
}

function start_recorder(
    runtime_dir: string,
    native_spawn_recorder: (
        backend: RecorderBackend,
        audio_path: string,
        log_path: string,
    ) -> NativeRecorder throws baml.errors.HostCallable,
    native_recorder_exists: (process: NativeRecorder) -> bool throws baml.errors.HostCallable,
) -> RecordingState {
    let session_dir = join_path(runtime_dir, `recording-${baml.id.new()}`);
    baml.fs.mkdir(session_dir, baml.fs.MkdirOptions { recursive: false });
    let audio_path = join_path(session_dir, "recording.wav");
    let log_path = join_path(session_dir, "recording.stderr.log");

    for (let backend in [RecorderBackend.PwRecord, RecorderBackend.Ffmpeg]) {
        let state = try_start_recorder(backend, audio_path, log_path, session_dir, native_spawn_recorder, native_recorder_exists);
        if let started: RecordingState = state {
            return started;
        }
    }
    baml.fs.remove_dir_all(session_dir) catch_all (error) {
        _ => null
    };
    invalid_argument("Could not start audio capture; install pw-record or ffmpeg")
}

function record_interactively(
    runtime_dir: string,
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
) -> RecordedAudio {
    let state = start_recorder(runtime_dir, native_spawn_recorder, native_recorder_exists);
    let _ = baml.io.input("Recording... Press Enter to stop.\n");
    native_stop_recorder(state.process, state.backend);
    finish_recording(state)
}

function sway_start_recording(
    runtime_dir: string,
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
) -> null {
    let state_path = recording_state_path(runtime_dir);
    if let old: RecordingState = read_recording_state(state_path) {
        if (native_recorder_exists(old.process)) {
            invalid_argument("Sway recording is already active")
        }
        cleanup_recording_files(state_path, old)
    }

    let state = start_recorder(runtime_dir, native_spawn_recorder, native_recorder_exists);
    commit_recording_state(state_path, state) catch_all (error) {
        _ => {
            native_stop_recorder(state.process, state.backend) catch_all (stop_error) {
                _ => null
            };
            cleanup_recording_files(state_path, state);
            throw error;
        },
    };
    null
}

function sway_stop_recording(
    runtime_dir: string,
    native_recorder_exists: (process: NativeRecorder) -> bool throws baml.errors.HostCallable,
    native_stop_recorder: (
        process: NativeRecorder,
        backend: RecorderBackend,
    ) -> null throws baml.errors.HostCallable,
) -> RecordedAudio {
    let state_path = recording_state_path(runtime_dir);
    let state = read_recording_state(state_path) ?? invalid_argument("No active Sway recording");
    if (!native_recorder_exists(state.process)) {
        cleanup_recording_files(state_path, state);
        invalid_argument("Sway recording process is not running anymore")
    }
    native_stop_recorder(state.process, state.backend);
    remove_file_if_present(state_path);
    finish_recording(state)
}

function sway_cancel_recording(
    runtime_dir: string,
    native_recorder_exists: (process: NativeRecorder) -> bool throws baml.errors.HostCallable,
    native_stop_recorder: (
        process: NativeRecorder,
        backend: RecorderBackend,
    ) -> null throws baml.errors.HostCallable,
) -> null {
    let state_path = recording_state_path(runtime_dir);
    if let state: RecordingState = read_recording_state(state_path) {
        if (native_recorder_exists(state.process)) {
            native_stop_recorder(state.process, state.backend)
        }
        cleanup_recording_files(state_path, state)
    }
    null
}

test "recording state round trips through JSON" {
    let state = RecordingState {
        process: NativeRecorder { pid: 42, started_at: 100 },
        backend: RecorderBackend.PwRecord,
        audio_path: "/tmp/audio.wav",
        session_dir: "/tmp/session",
    };
    assert.equal(baml.json.from_string<RecordingState>(baml.json.to_string(state)), state)
}
