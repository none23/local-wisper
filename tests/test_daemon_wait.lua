local repo_root = vim.fn.getcwd()
local fixture_root = repo_root .. "/tests/fixtures/slow_daemon"

local tmp_root = vim.fn.tempname()
vim.fn.mkdir(tmp_root, "p")
vim.env.XDG_CACHE_HOME = tmp_root .. "/cache"
vim.env.XDG_DATA_HOME = tmp_root .. "/data"
vim.fn.mkdir(vim.env.XDG_CACHE_HOME, "p")
vim.fn.mkdir(vim.env.XDG_DATA_HOME, "p")

vim.opt.runtimepath:prepend(repo_root)
vim.opt.runtimepath:prepend(fixture_root)

local messages = {}
local original_echo = vim.api.nvim_echo

vim.api.nvim_echo = function(chunks, history, opts)
  local parts = {}
  for _, chunk in ipairs(chunks) do
    table.insert(parts, chunk[1])
  end
  table.insert(messages, table.concat(parts))
  return original_echo(chunks, history, opts)
end

vim.api.nvim_buf_set_lines(0, 0, -1, false, { "start" })

local lw = require("lw")
lw.setup({
  python_bin = repo_root .. "/.venv/bin/python",
  preload_on_setup = false,
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

local inserted = vim.wait(14000, function()
  local lines = vim.api.nvim_buf_get_lines(0, 0, -1, false)
  for _, line in ipairs(lines) do
    if line == "slow daemon ok" then
      return true
    end
  end
  return false
end, 50)

if not inserted then
  error("expected transcript insertion; messages: " .. table.concat(messages, " | "))
end

for _, message in ipairs(messages) do
  if message:find("daemon did not become ready", 1, true) then
    error("unexpected readiness timeout; messages: " .. table.concat(messages, " | "))
  end
end

print("daemon wait regression test passed")
