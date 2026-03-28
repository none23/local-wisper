if vim.g.loaded_lw_plugin == 1 then
  return
end
vim.g.loaded_lw_plugin = 1

vim.api.nvim_create_user_command("LW", function()
  require("lw").toggle()
end, { desc = "Local speech record/transcribe and insert below cursor" })

vim.api.nvim_create_user_command("LWInstallDeps", function()
  require("lw").install_deps()
end, { desc = "Install lw.nvim Python dependencies" })
