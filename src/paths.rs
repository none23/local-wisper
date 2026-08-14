use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

const MODEL_DIR_NAME: &str = "parakeet-tdt-0.6b-v3-fp16-f88260fa";

pub fn cache_root() -> Result<PathBuf> {
    let root = if let Some(path) = env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(path)
    } else if let Some(home) = env::var_os("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        bail!("HOME or XDG_CACHE_HOME is required")
    };
    Ok(root.join("local-wisper"))
}

pub fn model_dir() -> Result<PathBuf> {
    Ok(cache_root()?.join("models").join(MODEL_DIR_NAME))
}

pub fn runtime_dir() -> Result<PathBuf> {
    let path = match env::var_os("XDG_RUNTIME_DIR") {
        Some(root) => PathBuf::from(root).join("local-wisper"),
        None => cache_root()?.join("runtime"),
    };
    fs::create_dir_all(&path)
        .with_context(|| format!("failed to create runtime directory {}", path.display()))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("failed to secure runtime directory {}", path.display()))?;
    Ok(path)
}

pub fn socket_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("daemon.sock"))
}

pub fn daemon_lock_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("daemon.lock"))
}

pub fn daemon_error_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("daemon.error"))
}

pub fn daemon_log_path() -> Result<PathBuf> {
    Ok(cache_root()?.join("daemon.log"))
}
