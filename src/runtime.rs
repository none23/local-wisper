use std::env;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use libloading::os::unix::Library;

static CUDNN: OnceLock<Result<Vec<Library>, String>> = OnceLock::new();

const CUDNN_LIBRARIES: &[&str] = &[
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
];

pub fn prepare_cuda() -> Result<()> {
    CUDNN
        .get_or_init(|| load_cudnn().map_err(|error| format!("{error:#}")))
        .as_ref()
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!(error.clone()))
}

fn load_cudnn() -> Result<Vec<Library>> {
    let directory = candidate_directories()
        .into_iter()
        .find(|directory| directory.join(CUDNN_LIBRARIES[0]).is_file())
        .context("cuDNN 9 was not found; rerun install.sh or install the system cudnn package")?;
    let mut libraries = Vec::with_capacity(CUDNN_LIBRARIES.len());
    for name in CUDNN_LIBRARIES {
        let path = directory.join(name);
        if !path.is_file() {
            bail!("incomplete cuDNN installation: missing {}", path.display())
        }
        let library = unsafe { Library::open(Some(&path), libc::RTLD_NOW | libc::RTLD_GLOBAL) }
            .with_context(|| format!("failed to load {}", path.display()))?;
        libraries.push(library);
    }
    Ok(libraries)
}

fn candidate_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Some(path) = env::var_os("LW_RUNTIME_LIB_DIR") {
        directories.push(PathBuf::from(path));
    }
    if let Some(path) = installed_library_dir() {
        directories.push(path);
    }
    directories.extend([
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/local/cuda/lib64"),
    ]);
    directories
}

fn installed_library_dir() -> Option<PathBuf> {
    let executable = env::current_exe().ok()?;
    let prefix = executable.parent()?.parent()?;
    Some(prefix.join(Path::new("lib/local-wisper")))
}
