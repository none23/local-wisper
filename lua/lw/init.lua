local M = {}

M.config = {
  python_bin = nil,
}

local state = {
  recording = false,
  audio_path = nil,
  record_job = nil,
  stop_map_set = false,
  transcribe_done = false,
}

function M.setup(opts)
  M.config = vim.tbl_extend("force", M.config, opts or {})
end

local function status(text, hl)
  vim.api.nvim_echo({ { text, hl or "None" } }, false, {})
end

local function clear_stop_mapping()
  if not state.stop_map_set then
    return
  end
  pcall(vim.keymap.del, "n", "<CR>")
  state.stop_map_set = false
end

local function add_text_below_cursor(text)
  local row = vim.api.nvim_win_get_cursor(0)[1]
  vim.api.nvim_buf_set_lines(0, row, row, false, vim.split(text, "\n", { plain = true }))
end

local function transcribe_and_insert()
  local script = vim.api.nvim_get_runtime_file("scripts/transcribe_file.py", false)[1]
  if not script or script == "" then
    status("LW: could not find scripts/transcribe_file.py", "ErrorMsg")
    return
  end
  local repo_root = vim.fn.fnamemodify(script, ":h:h")
  local venv_python = repo_root .. "/.venv/bin/python"
  local python_bin = M.config.python_bin
  if python_bin == nil or python_bin == "" then
    python_bin = vim.fn.executable(venv_python) == 1 and venv_python or "python3"
  end
  local stderr_lines = {}

  local python_cmd = {
    python_bin,
    script,
    state.audio_path,
  }
  state.transcribe_done = false

  vim.fn.jobstart(python_cmd, {
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
        vim.schedule(function()
          status("LW error: " .. msg, "ErrorMsg")
        end)
      end
    end,
  })
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
  vim.keymap.set("n", "<CR>", function()
    M.stop()
  end, { silent = true, nowait = true, desc = "LW stop recording" })
  state.stop_map_set = true
end

function M.start()
  if state.recording then
    status("LW: already recording (press Enter to stop)", "WarningMsg")
    return
  end

  state.audio_path = vim.fn.tempname() .. ".wav"
  local cmd = {
    "pw-record",
    "--rate",
    "16000",
    "--channels",
    "1",
    "--format",
    "s16",
    state.audio_path,
  }

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
    status("LW: failed to start recorder (need pw-record)", "ErrorMsg")
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
