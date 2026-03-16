local M = {}
local DAEMON_READY_TIMEOUT_MS = 120000

M.config = {
  python_bin = nil,
  venv_dir = nil,
  auto_install_deps = true,
  model = "small",
  compute_type = "int8",
  device = "cpu",
  vad_filter = true,
  sample_rate = 16000,
  recorder_cmd = nil,
  preload_on_setup = true,
}

local state = {
  recording = false,
  audio_path = nil,
  record_job = nil,
  stop_map_set = false,
  stop_bufnr = nil,
  bootstrap_running = false,
  resolved_python = nil,
  daemon_config_key = nil,
  daemon_socket_path = nil,
  daemon_last_start_ms = 0,
  request_busy = false,
  request_id = 0,
  pending_request_id = nil,
  request_channel = nil,
  daemon_job = nil,
  daemon_start_error = nil,
  daemon_stderr_tail = {},
}

local function status(text, hl)
  vim.api.nvim_echo({ { text, hl or "None" } }, false, {})
end

local ensure_daemon
local daemon_reachable

local function daemon_script_and_repo_root()
  local script = vim.api.nvim_get_runtime_file("scripts/transcribe_daemon.py", false)[1]
  if not script or script == "" then
    return nil, nil
  end
  local repo_root = vim.fn.fnamemodify(script, ":h:h")
  return script, repo_root
end

local function default_venv_dir()
  return M.config.venv_dir or (vim.fn.stdpath("data") .. "/lw.nvim/.venv")
end

local function default_venv_python()
  return default_venv_dir() .. "/bin/python"
end

local function resolve_python_bin()
  if M.config.python_bin and M.config.python_bin ~= "" then
    return M.config.python_bin
  end
  if state.resolved_python and state.resolved_python ~= "" then
    return state.resolved_python
  end

  local venv_python = default_venv_python()
  if vim.fn.executable(venv_python) == 1 then
    state.resolved_python = venv_python
    return venv_python
  end

  local _, repo_root = daemon_script_and_repo_root()
  if repo_root then
    local repo_venv = repo_root .. "/.venv/bin/python"
    if vim.fn.executable(repo_venv) == 1 then
      state.resolved_python = repo_venv
      return repo_venv
    end
  end

  return venv_python
end

local function daemon_config_key()
  return table.concat({ M.config.model, M.config.compute_type, M.config.device, tostring(M.config.vad_filter) }, "|")
end

local function short_hash(text)
  local ok, digest = pcall(vim.fn.sha256, text)
  if ok and type(digest) == "string" and #digest >= 12 then
    return string.sub(digest, 1, 12)
  end
  return "default"
end

local function daemon_socket_path_for_key(key)
  local base = vim.fn.stdpath("cache") .. "/lw.nvim"
  local ok = pcall(vim.fn.mkdir, base, "p")
  if not ok then
    base = "/tmp/lw.nvim"
    pcall(vim.fn.mkdir, base, "p")
  end
  return base .. "/daemon-" .. short_hash(key) .. ".sock"
end

local function daemon_ready_max_attempts()
  return math.max(1, math.ceil(DAEMON_READY_TIMEOUT_MS / 150))
end

local function reset_daemon_start_state()
  state.daemon_start_error = nil
  state.daemon_stderr_tail = {}
end

local function daemon_start_failure_detail(exit_code)
  local detail = nil
  for i = #state.daemon_stderr_tail, 1, -1 do
    local line = state.daemon_stderr_tail[i]
    if line and line ~= "" then
      detail = line
      break
    end
  end
  if detail and detail ~= "" then
    return detail
  end
  return "exit code " .. tostring(exit_code)
end

local function add_text_below_cursor(text)
  local row = vim.api.nvim_win_get_cursor(0)[1]
  vim.api.nvim_buf_set_lines(0, row, row, false, vim.split(text, "\n", { plain = true }))
end

local function close_request_channel()
  if state.request_channel and state.request_channel > 0 then
    pcall(vim.fn.chanclose, state.request_channel)
  end
  state.request_channel = nil
end

local function clear_request_state()
  close_request_channel()
  state.request_busy = false
  state.pending_request_id = nil
end

function M.setup(opts)
  M.config = vim.tbl_extend("force", M.config, opts or {})

  if M.config.preload_on_setup then
    vim.schedule(function()
      local python_bin = resolve_python_bin()
      if vim.fn.executable(python_bin) == 1 then
        ensure_daemon()
      end
    end)
  end
end

function M.install_deps(cb)
  if state.bootstrap_running then
    status("LW: dependency install already running", "WarningMsg")
    return
  end

  local _, repo_root = daemon_script_and_repo_root()
  if not repo_root then
    status("LW: could not find plugin files", "ErrorMsg")
    return
  end

  local req = repo_root .. "/requirements.txt"
  if vim.fn.filereadable(req) ~= 1 then
    status("LW: requirements.txt not found", "ErrorMsg")
    return
  end

  local venv_dir = default_venv_dir()
  local venv_python = default_venv_python()
  vim.fn.mkdir(vim.fn.fnamemodify(venv_dir, ":h"), "p")

  local cmd = "python3 -m venv "
    .. vim.fn.shellescape(venv_dir)
    .. " && "
    .. vim.fn.shellescape(venv_python)
    .. " -m pip install -U pip && "
    .. vim.fn.shellescape(venv_python)
    .. " -m pip install -r "
    .. vim.fn.shellescape(req)

  state.bootstrap_running = true
  vim.notify("lw.nvim: installing Python dependencies...", vim.log.levels.INFO)

  local job = vim.fn.jobstart({ "sh", "-c", cmd }, {
    on_exit = function(_, code, _)
      state.bootstrap_running = false
      vim.schedule(function()
        if code == 0 then
          state.resolved_python = venv_python
          vim.notify("lw.nvim: dependencies installed", vim.log.levels.INFO)
          if cb then
            cb(true)
          end
        else
          vim.notify("lw.nvim: dependency install failed (exit " .. code .. ")", vim.log.levels.ERROR)
          if cb then
            cb(false)
          end
        end
      end)
    end,
  })

  if job <= 0 then
    state.bootstrap_running = false
    status("LW: failed to start dependency install", "ErrorMsg")
  end
end

local function ensure_python_ready()
  local py = resolve_python_bin()
  if vim.fn.executable(py) == 1 then
    return true
  end

  if M.config.python_bin and M.config.python_bin ~= "" then
    status("LW: python binary not executable: " .. py, "ErrorMsg")
    return false
  end

  if M.config.auto_install_deps then
    M.install_deps()
    status("LW: installing dependencies, run :LW again when done", "WarningMsg")
    return false
  end

  status("LW: missing Python deps. Run :LWInstallDeps", "ErrorMsg")
  return false
end

local function clear_stop_mapping()
  if not state.stop_map_set then
    return
  end
  if state.stop_bufnr and vim.api.nvim_buf_is_valid(state.stop_bufnr) then
    pcall(vim.keymap.del, "n", "<CR>", { buffer = state.stop_bufnr })
  end
  state.stop_map_set = false
  state.stop_bufnr = nil
end

local function start_daemon()
  local script, repo_root = daemon_script_and_repo_root()
  if not script then
    status("LW: could not find scripts/transcribe_daemon.py", "ErrorMsg")
    return false
  end

  local python_bin = resolve_python_bin()
  if vim.fn.executable(python_bin) ~= 1 then
    status("LW: python binary not executable: " .. python_bin, "ErrorMsg")
    return false
  end

  local key = daemon_config_key()
  if state.daemon_config_key ~= key then
    state.daemon_config_key = key
    state.daemon_socket_path = daemon_socket_path_for_key(key)
  end
  local now_ms = vim.loop.hrtime() / 1000000
  if now_ms - state.daemon_last_start_ms < 1000 then
    return true
  end

  reset_daemon_start_state()
  local cmd = {
    python_bin,
    script,
    "--model",
    M.config.model,
    "--compute-type",
    M.config.compute_type,
    "--device",
    M.config.device,
    "--socket",
    state.daemon_socket_path,
  }
  if M.config.vad_filter then
    table.insert(cmd, "--vad-filter")
  else
    table.insert(cmd, "--no-vad-filter")
  end

  local job = vim.fn.jobstart(cmd, {
    cwd = repo_root,
    detach = true,
    stderr_buffered = false,
    on_stderr = function(_, data, _)
      if not data then
        return
      end
      for _, line in ipairs(data) do
        if line and line ~= "" then
          table.insert(state.daemon_stderr_tail, line)
          if #state.daemon_stderr_tail > 20 then
            table.remove(state.daemon_stderr_tail, 1)
          end
        end
      end
    end,
    on_exit = function(_, code, _)
      vim.schedule(function()
        state.daemon_job = nil
        if code ~= 0 and not daemon_reachable() then
          state.daemon_start_error = daemon_start_failure_detail(code)
        end
      end)
    end,
  })
  if job <= 0 then
    status("LW: failed to start transcription daemon", "ErrorMsg")
    return false
  end

  state.daemon_job = job
  state.daemon_last_start_ms = now_ms
  return true
end

daemon_reachable = function()
  if not state.daemon_socket_path or state.daemon_socket_path == "" then
    return false
  end

  local ok, chan = pcall(vim.fn.sockconnect, "pipe", state.daemon_socket_path, { rpc = false })
  if not ok then
    return false
  end
  if chan <= 0 then
    return false
  end
  pcall(vim.fn.chanclose, chan)
  return true
end

ensure_daemon = function()
  local key = daemon_config_key()
  if state.daemon_config_key ~= key or not state.daemon_socket_path then
    state.daemon_config_key = key
    state.daemon_socket_path = daemon_socket_path_for_key(key)
  end

  if daemon_reachable() then
    state.daemon_start_error = nil
    return true
  end

  return start_daemon()
end

local function handle_daemon_message(line)
  local ok, msg = pcall(vim.json.decode, line)
  if not ok or type(msg) ~= "table" then
    return
  end

  if msg.id ~= state.pending_request_id then
    return
  end

  clear_request_state()

  if msg.type == "result" and type(msg.text) == "string" and msg.text ~= "" then
    add_text_below_cursor(msg.text)
    status("LW: inserted transcript", "Question")
    return
  end

  if msg.type == "no_speech" then
    status("LW: no speech detected", "WarningMsg")
    return
  end

  if msg.type == "error" then
    local detail = msg.error or "unknown error"
    status("LW: transcription failed: " .. detail, "ErrorMsg")
    return
  end

  status("LW: unexpected daemon response", "ErrorMsg")
end

local function send_transcribe_request(audio_path, attempt)
  if state.request_busy then
    status("LW: transcription already running", "WarningMsg")
    return false
  end

  attempt = attempt or 0
  local max_attempts = daemon_ready_max_attempts()
  if not ensure_daemon() then
    return false
  end

  if not daemon_reachable() then
    if state.daemon_start_error then
      status("LW: daemon failed to start: " .. state.daemon_start_error, "ErrorMsg")
      return false
    end
    if attempt == 0 then
      status("LW: loading model...", "ModeMsg")
    end
    if attempt >= max_attempts then
      status("LW: daemon did not become ready", "ErrorMsg")
      return false
    end
    vim.defer_fn(function()
      send_transcribe_request(audio_path, attempt + 1)
    end, 150)
    return true
  end

  state.request_id = state.request_id + 1
  state.pending_request_id = state.request_id

  local ok, chan = pcall(vim.fn.sockconnect, "pipe", state.daemon_socket_path, {
    rpc = false,
    on_data = function(_, data, _)
      if not data then
        return
      end
      for _, line in ipairs(data) do
        if line and line ~= "" then
          vim.schedule(function()
            handle_daemon_message(line)
          end)
        end
      end
    end,
  })
  if not ok then
    chan = -1
  end

  if chan <= 0 then
    if attempt >= max_attempts then
      status("LW: failed to connect to daemon", "ErrorMsg")
      return false
    end
    vim.defer_fn(function()
      send_transcribe_request(audio_path, attempt + 1)
    end, 150)
    return true
  end

  state.request_busy = true
  state.request_channel = chan

  local payload = vim.json.encode({
    type = "transcribe",
    id = state.pending_request_id,
    audio_path = audio_path,
  })
  vim.fn.chansend(chan, payload .. "\n")
  pcall(vim.fn.chanclose, chan, "stdin")

  local pending_id = state.pending_request_id
  vim.defer_fn(function()
    if state.pending_request_id == pending_id then
      clear_request_state()
      status("LW: transcription timed out", "ErrorMsg")
    end
  end, 120000)

  return true
end

local function transcribe_and_insert()
  send_transcribe_request(state.audio_path, 0)
end

function M.stop()
  if not state.recording then
    return
  end

  state.recording = false
  clear_stop_mapping()

  if state.record_job then
    pcall(vim.fn.jobstop, state.record_job)
    state.record_job = nil
  end

  status("LW: transcribing...", "ModeMsg")
  transcribe_and_insert()
end

local function set_stop_mapping()
  if state.stop_map_set then
    return
  end

  local bufnr = vim.api.nvim_get_current_buf()
  state.stop_bufnr = bufnr
  vim.keymap.set("n", "<CR>", function()
    M.stop()
  end, { buffer = bufnr, silent = true, nowait = true, desc = "LW stop recording" })
  state.stop_map_set = true
end

local function build_record_cmd(audio_path)
  if type(M.config.recorder_cmd) == "table" and #M.config.recorder_cmd > 0 then
    local cmd = vim.deepcopy(M.config.recorder_cmd)
    table.insert(cmd, audio_path)
    return cmd
  end

  return {
    "pw-record",
    "--rate",
    tostring(M.config.sample_rate),
    "--channels",
    "1",
    "--format",
    "s16",
    audio_path,
  }
end

function M.start()
  if state.recording then
    status("LW: already recording (press Enter to stop)", "WarningMsg")
    return
  end

  if not ensure_python_ready() then
    return
  end

  ensure_daemon()

  state.audio_path = vim.fn.tempname() .. ".wav"
  local cmd = build_record_cmd(state.audio_path)

  state.record_job = vim.fn.jobstart(cmd, {
    detach = false,
    on_exit = function(_, code, _)
      if state.recording and code ~= 0 then
        state.recording = false
        clear_stop_mapping()
        vim.schedule(function()
          status("LW: recorder exited unexpectedly", "ErrorMsg")
        end)
      end
    end,
  })

  if state.record_job <= 0 then
    status("LW: failed to start recorder (need pw-record or configured recorder_cmd)", "ErrorMsg")
    return
  end

  state.recording = true
  set_stop_mapping()
  status("recording (press Enter to stop)", "ModeMsg")
end

function M.toggle()
  if state.recording then
    M.stop()
    return
  end
  M.start()
end

return M
