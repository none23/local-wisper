use std::fs;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

pub fn runtime_dir() -> Result<PathBuf> {
    let uid = unsafe { libc::geteuid() };
    let system_runtime = PathBuf::from(format!("/run/user/{uid}"));
    let path = if system_runtime.is_dir() {
        system_runtime.join("local-wisper")
    } else {
        PathBuf::from(format!("/tmp/local-wisper-{uid}"))
    };
    match fs::create_dir(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to create runtime directory {}", path.display()));
        }
    }
    let metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("failed to inspect runtime directory {}", path.display()))?;
    if !metadata.is_dir() || metadata.uid() != uid {
        bail!(
            "runtime path {} is not a directory owned by user {uid}",
            path.display()
        )
    }
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("failed to secure runtime directory {}", path.display()))?;
    Ok(path)
}

pub fn daemon_lock_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("daemon.lock"))
}
