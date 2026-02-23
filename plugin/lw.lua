if vim.g.loaded_lw_plugin == 1 then
  return
end
vim.g.loaded_lw_plugin = 1

vim.api.nvim_create_user_command("LW", function()
  require("lw").toggle()
end, { desc = "Local Whisper record/transcribe and insert below cursor" })
