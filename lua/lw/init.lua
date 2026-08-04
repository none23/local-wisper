-- Keep the standard Neovim module path stable while the implementation lives
-- with the repository's optional integrations.
local implementation = vim.api.nvim_get_runtime_file("integrations/neovim/lua/lw/init.lua", false)[1]

if not implementation or implementation == "" then
  error("lw.nvim: Neovim integration implementation not found", 0)
end

return dofile(implementation)
