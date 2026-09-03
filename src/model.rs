use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use parakeet_rs::{ExecutionConfig, ParakeetUnified, TimestampMode, Transcriber};

use crate::{paths, runtime};

struct Model {
    inner: ParakeetUnified,
}

#[derive(Default)]
pub struct ModelHost {
    lock: Mutex<Option<File>>,
    model: Mutex<Option<Model>>,
}

impl ModelHost {
    pub fn acquire_lock(&self) -> Result<bool> {
        let lock_path = paths::daemon_lock_path()?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("failed to open model lock {}", lock_path.display()))?;
        match file.try_lock_exclusive() {
            Ok(()) => {
                *self
                    .lock
                    .lock()
                    .map_err(|_| anyhow::anyhow!("model lock holder was poisoned"))? = Some(file);
                Ok(true)
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(false),
            Err(error) => Err(error).context("failed to acquire the per-user model lock"),
        }
    }

    pub fn load(&self, model_dir: String, variant: baml_sdk::ModelVariant) -> Result<()> {
        if self
            .lock
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock holder was poisoned"))?
            .is_none()
        {
            bail!("refusing to load Parakeet Unified without the per-user model lock")
        }
        let mut slot = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model holder was poisoned"))?;
        if slot.is_some() {
            bail!("the resident model is already loaded")
        }
        let (variant_name, device, config) = match variant {
            baml_sdk::ModelVariant::Fp32 => {
                runtime::prepare_cuda()?;
                ("FP32", "CUDA", strict_cuda_config())
            }
            baml_sdk::ModelVariant::Int8 => ("INT8", "CPU", ExecutionConfig::new()),
        };
        let started = Instant::now();
        let inner = ParakeetUnified::from_pretrained(Path::new(&model_dir), Some(config))
            .with_context(|| {
                format!(
                    "failed to load Parakeet Unified {variant_name} with the {device} execution provider from {model_dir}"
                )
            })?;
        eprintln!(
            "Parakeet Unified {variant_name} loaded on {device} in {:.2?}",
            started.elapsed()
        );
        *slot = Some(Model { inner });
        Ok(())
    }

    pub fn transcribe(&self, audio_path: String) -> Result<String> {
        let mut slot = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model holder was poisoned"))?;
        let model = slot.as_mut().context("resident model is not loaded")?;
        let started = Instant::now();
        let result = Transcriber::transcribe_file(
            &mut model.inner,
            Path::new(&audio_path),
            Some(TimestampMode::Sentences),
        )
        .with_context(|| format!("failed to transcribe {audio_path}"))?;
        eprintln!("transcribed {audio_path} in {:.2?}", started.elapsed());
        Ok(result.text.trim().to_owned())
    }
}

fn strict_cuda_config() -> ExecutionConfig {
    ExecutionConfig::new().with_custom_configure(|builder| {
        Ok(builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_load_requires_the_per_user_lock() {
        let error = ModelHost::default()
            .load("/does/not/exist".to_owned(), baml_sdk::ModelVariant::Int8)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without the per-user model lock")
        );
    }
}
