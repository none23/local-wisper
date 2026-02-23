local M = {}

M.config = {
  python_bin = nil,
  venv_dir = nil,
  auto_install_deps = true,
  model = "small.en",
  compute_type = "int8",
  sample_rate = 16000,
  recorder_cmd = nil,
}

local state = {
  recording = false,
  audio_path = nil,
  record_job = nil,
  stop_map_set = false,
  stop_bufnr = nil,
  transcribe_done = false,
  bootstrap_running = false,
  resolved_python = nil,
}

function M.setup(opts)
  M.config = vim.tbl_extend("force", M.config, opts or {})
end

local function status(text, hl)
  vim.api.nvim_echo({ { text, hl or "None" } }, false, {})
end

local function script_and_repo_root()
  local script = vim.api.nvim_get_runtime_file("scripts/transcribe_file.py", false)[1]
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

  local _, repo_root = script_and_repo_root()
  if repo_root then
    local repo_venv = repo_root .. "/.venv/bin/python"
    if vim.fn.executable(repo_venv) == 1 then
      state.resolved_python = repo_venv
      return repo_venv
    end
  end

  return venv_python
end

function M.install_deps(cb)
  if state.bootstrap_running then
    status("LW: dependency install already running", "WarningMsg")
    return
  end

  local _, repo_root = script_and_repo_root()
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

local function add_text_below_cursor(text)
  local row = vim.api.nvim_win_get_cursor(0)[1]
  vim.api.nvim_buf_set_lines(0, row, row, false, vim.split(text, "\n", { plain = true }))
end

local function transcribe_and_insert()
  local script = script_and_repo_root()
  if not script then
    status("LW: could not find scripts/transcribe_file.py", "ErrorMsg")
    return
  end

  local python_bin = resolve_python_bin()
  if vim.fn.executable(python_bin) ~= 1 then
    status("LW: python binary not executable: " .. python_bin, "ErrorMsg")
    return
  end

  local stderr_lines = {}
  local python_cmd = {
    python_bin,
    script,
    state.audio_path,
    "--model",
    M.config.model,
    "--compute-type",
    M.config.compute_type,
  }

  state.transcribe_done = false
  local transcribe_job = vim.fn.jobstart(python_cmd, {
    stdout_buffered = true,
    stderr_buffered = true,
    on_exit = function(_, code, _)
      vim.schedule(function()
        if state.transcribe_done then
          return
        end
        if code == 2 then
          status("LW: no speech detected", "WarningMsg")
          return
        end
        if code ~= 0 then
          local extra = ""
          if #stderr_lines > 0 then
            extra = ": " .. stderr_lines[1]
          end
          status("LW: transcription failed" .. extra, "ErrorMsg")
        end
      end)
    end,
    on_stdout = function(_, data, _)
      if not data then
        return
      end
      local lines = {}
      for _, line in ipairs(data) do
        if line ~= nil and line ~= "" then
          table.insert(lines, line)
        end
      end
      if #lines == 0 then
        return
      end
      local text = table.concat(lines, "\n")
      state.transcribe_done = true
      vim.schedule(function()
        add_text_below_cursor(text)
        status("LW: inserted transcript", "Question")
      end)
    end,
    on_stderr = function(_, data, _)
      if not data then
        return
      end
      local msg = table.concat(data, "\n"):gsub("%s+$", "")
      if msg ~= "" then
        table.insert(stderr_lines, msg)
      end
    end,
  })

  if transcribe_job <= 0 then
    status("LW: failed to start transcription process", "ErrorMsg")
  end
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
