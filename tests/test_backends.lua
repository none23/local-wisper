local repo_root = vim.fn.getcwd()
local fixture_root = repo_root .. "/tests/fixtures/backend_echo"

local tmp_root = vim.fn.tempname()
vim.fn.mkdir(tmp_root, "p")
vim.env.XDG_CACHE_HOME = tmp_root .. "/cache"
vim.env.XDG_DATA_HOME = tmp_root .. "/data"
vim.fn.mkdir(vim.env.XDG_CACHE_HOME, "p")
vim.fn.mkdir(vim.env.XDG_DATA_HOME, "p")

vim.opt.runtimepath:prepend(repo_root)
vim.opt.runtimepath:prepend(fixture_root)

vim.api.nvim_buf_set_lines(0, 0, -1, false, { "start" })

local lw = require("lw")

local cases = {
  {
    backend = "parakeet",
    model = "nvidia/parakeet-tdt-0.6b-v3",
    compute_type = "float16",
    device = "cuda",
    vad_filter = false,
  },
  {
    backend = "whisper",
    model = "small",
    compute_type = "int8",
    device = "cpu",
    vad_filter = true,
  },
}

for _, case in ipairs(cases) do
  lw.setup({
    python_bin = repo_root .. "/.venv/bin/python",
    preload_on_setup = false,
    backend = case.backend,
    model = case.model,
    compute_type = case.compute_type,
    device = case.device,
    vad_filter = case.vad_filter,
    recorder_cmd = {
      "sh",
      "-c",
      "dd if=/dev/zero bs=4096 count=1 of=\"$1\" >/dev/null 2>&1; sleep 60",
      "lw-test-recorder",
    },
  })

  lw.start()
  vim.wait(250)
  lw.stop()

  local expected = table.concat({
    case.backend,
    case.model,
    case.compute_type,
    case.device,
    case.vad_filter and "true" or "false",
  }, "|")

  local inserted = vim.wait(5000, function()
    local lines = vim.api.nvim_buf_get_lines(0, 0, -1, false)
    for _, line in ipairs(lines) do
      if line == expected then
        return true
      end
    end
    return false
  end, 50)

  if not inserted then
    error("expected transcript insertion for " .. expected)
  end
end

print("backend regression test passed")
