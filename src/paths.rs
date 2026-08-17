use std::fs;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub fn runtime_dir() -> Result<PathBuf> {
    let uid = unsafe { libc::geteuid() };
    let system_runtime = PathBuf::from(format!("/run/user/{uid}"));
    let path = if system_runtime.is_dir() {
        system_runtime.join("local-wisper")
    } else {
        PathBuf::from(format!("/tmp/local-wisper-{uid}"))
    };
    secure_user_dir(&path, uid)?;
    Ok(path)
}

pub fn data_dir() -> Result<PathBuf> {
    let root = data_root(
        std::env::var_os("XDG_DATA_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from),
        std::env::var_os("HOME").map(PathBuf::from),
    )?;
    let path = root.join("local-wisper");
    let uid = unsafe { libc::geteuid() };
    secure_user_dir(&path, uid)?;
    Ok(path)
}

fn data_root(xdg_data_home: Option<PathBuf>, home: Option<PathBuf>) -> Result<PathBuf> {
    let root = xdg_data_home
        .or_else(|| home.map(|path| path.join(".local/share")))
        .context("HOME or XDG_DATA_HOME is required")?;
    if !root.is_absolute() {
        bail!("XDG_DATA_HOME or HOME must be an absolute path")
    }
    Ok(root)
}

fn secure_user_dir(path: &Path, uid: u32) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("failed to create directory {}", path.display()))?;
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect directory {}", path.display()))?;
    if !metadata.is_dir() || metadata.uid() != uid {
        bail!(
            "path {} is not a directory owned by user {uid}",
            path.display()
        )
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("failed to secure directory {}", path.display()))?;
    Ok(())
}

pub fn daemon_lock_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("daemon.lock"))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn secure_user_dir_creates_private_parent_directories() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "local-wisper-path-test-{}-{suffix}",
            std::process::id()
        ));
        let path = root.join("nested/local-wisper");

        secure_user_dir(&path, unsafe { libc::geteuid() }).unwrap();

        assert!(path.is_dir());
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn data_root_prefers_xdg_and_falls_back_to_home() {
        assert_eq!(
            data_root(
                Some(PathBuf::from("/xdg")),
                Some(PathBuf::from("/home/user"))
            )
            .unwrap(),
            PathBuf::from("/xdg")
        );
        assert_eq!(
            data_root(None, Some(PathBuf::from("/home/user"))).unwrap(),
            PathBuf::from("/home/user/.local/share")
        );
        assert!(data_root(Some(PathBuf::from("relative")), None).is_err());
        assert!(data_root(None, None).is_err());
    }
}
