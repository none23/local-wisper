use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

use crate::{paths, runtime};

const REVISION: &str = "f88260fa0777fe0868dda6df85d1a98f012a4a7a";
const REPOSITORY: &str = "ysdede/parakeet-tdt-0.6b-v3-onnx";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum DevicePreference {
    #[default]
    Auto,
    Cuda,
    Cpu,
}

impl DevicePreference {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda),
            "cpu" => Ok(Self::Cpu),
            _ => bail!("invalid daemon device {value}"),
        }
    }
}

struct Asset {
    remote_name: &'static str,
    local_name: &'static str,
    size: u64,
    sha256: &'static str,
}

struct Variant {
    name: &'static str,
    cache_dir: &'static str,
    assets: &'static [Asset],
}

const VOCAB: Asset = Asset {
    remote_name: "vocab.txt",
    local_name: "vocab.txt",
    size: 102_132,
    sha256: "ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742",
};

const FP16_ASSETS: &[Asset] = &[
    Asset {
        remote_name: "encoder-model.fp16.onnx",
        local_name: "encoder-model.onnx",
        size: 1_238_960_452,
        sha256: "a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717",
    },
    Asset {
        remote_name: "decoder_joint-model.fp16.onnx",
        local_name: "decoder_joint-model.onnx",
        size: 36_266_140,
        sha256: "b33a73b7c1d71b9d5a0911f5cb478be3dcbf79f53355c531ab1cd1dcd68ad8ef",
    },
    VOCAB,
];

const INT8_ASSETS: &[Asset] = &[
    Asset {
        remote_name: "encoder-model.int8.onnx",
        local_name: "encoder-model.onnx",
        size: 652_183_999,
        sha256: "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
    },
    Asset {
        remote_name: "decoder_joint-model.int8.onnx",
        local_name: "decoder_joint-model.onnx",
        size: 18_202_004,
        sha256: "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
    },
    VOCAB,
];

const FP16: Variant = Variant {
    name: "FP16",
    cache_dir: "parakeet-tdt-0.6b-v3-fp16-f88260fa",
    assets: FP16_ASSETS,
};

const INT8: Variant = Variant {
    name: "INT8",
    cache_dir: "parakeet-tdt-0.6b-v3-int8-f88260fa",
    assets: INT8_ASSETS,
};

pub struct Model {
    inner: ParakeetTDT,
}

impl Model {
    pub fn load(preference: DevicePreference) -> Result<Self> {
        match preference {
            DevicePreference::Cpu => load_cpu(),
            DevicePreference::Auto | DevicePreference::Cuda if runtime::cuda_hardware_present() => {
                load_cuda().or_else(|cuda_error| {
                    eprintln!(
                        "CUDA model initialization failed; falling back to CPU: {cuda_error:#}"
                    );
                    load_cpu()
                })
            }
            DevicePreference::Auto | DevicePreference::Cuda => {
                eprintln!("no NVIDIA CUDA device detected; using CPU");
                load_cpu()
            }
        }
    }

    pub fn transcribe(&mut self, audio: &Path) -> Result<String> {
        let started = Instant::now();
        let result = self
            .inner
            .transcribe_file(audio, Some(TimestampMode::Sentences))
            .with_context(|| format!("failed to transcribe {}", audio.display()))?;
        eprintln!(
            "transcribed {} in {:.2?}",
            audio.display(),
            started.elapsed()
        );
        Ok(result.text.trim().to_owned())
    }
}

fn load_cuda() -> Result<Model> {
    runtime::prepare_cuda()?;
    load_variant(&FP16, strict_cuda_config(), "CUDA")
}

fn load_cpu() -> Result<Model> {
    load_variant(&INT8, ExecutionConfig::new(), "CPU")
}

fn load_variant(variant: &Variant, config: ExecutionConfig, device: &str) -> Result<Model> {
    let model_dir = paths::model_dir(variant.cache_dir)?;
    prepare(&model_dir, variant)?;
    let started = Instant::now();
    let inner = ParakeetTDT::from_pretrained(&model_dir, Some(config)).with_context(|| {
        format!(
            "failed to load Parakeet {} with the {device} execution provider from {}",
            variant.name,
            model_dir.display()
        )
    })?;
    eprintln!(
        "Parakeet {} loaded on {device} in {:.2?}",
        variant.name,
        started.elapsed()
    );
    Ok(Model { inner })
}

fn prepare(model_dir: &Path, variant: &Variant) -> Result<()> {
    fs::create_dir_all(model_dir)
        .with_context(|| format!("failed to create model cache {}", model_dir.display()))?;
    let marker = model_dir.join(".complete");
    if marker.is_file()
        && variant
            .assets
            .iter()
            .all(|asset| has_expected_size(model_dir, asset))
    {
        return Ok(());
    }

    let client = Client::builder()
        .build()
        .context("failed to initialize the model download client")?;
    for asset in variant.assets {
        let destination = model_dir.join(asset.local_name);
        if has_expected_size(model_dir, asset) && verify_sha256(&destination, asset.sha256)? {
            continue;
        }
        download_asset(&client, model_dir, asset)?;
    }

    let marker_part = model_dir.join(".complete.part");
    fs::write(
        &marker_part,
        format!("{REPOSITORY}@{REVISION} {}\n", variant.name),
    )
    .context("failed to write model completion marker")?;
    fs::rename(&marker_part, &marker).context("failed to commit model completion marker")?;
    Ok(())
}

fn strict_cuda_config() -> ExecutionConfig {
    ExecutionConfig::new().with_custom_configure(|builder| {
        Ok(builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?)
    })
}

fn has_expected_size(model_dir: &Path, asset: &Asset) -> bool {
    fs::metadata(model_dir.join(asset.local_name))
        .map(|metadata| metadata.len() == asset.size)
        .unwrap_or(false)
}

fn download_asset(client: &Client, model_dir: &Path, asset: &Asset) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{}",
        asset.remote_name
    );
    eprintln!("downloading {}", asset.remote_name);
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("failed to download {}", asset.remote_name))?
        .error_for_status()
        .with_context(|| format!("model server rejected {}", asset.remote_name))?;

    let part = model_dir.join(format!("{}.part", asset.local_name));
    let mut output = File::create(&part)
        .with_context(|| format!("failed to create partial model file {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut written = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = response
            .read(&mut buffer)
            .with_context(|| format!("failed while downloading {}", asset.remote_name))?;
        if count == 0 {
            break;
        }
        output.write_all(&buffer[..count])?;
        hasher.update(&buffer[..count]);
        written += count as u64;
    }
    output.sync_all()?;

    let digest = hex::encode(hasher.finalize());
    if written != asset.size || digest != asset.sha256 {
        bail!(
            "downloaded {} failed verification: expected {} bytes and {}, got {} bytes and {}",
            asset.remote_name,
            asset.size,
            asset.sha256,
            written,
            digest
        );
    }
    fs::rename(&part, model_dir.join(asset.local_name))
        .with_context(|| format!("failed to commit {}", asset.local_name))?;
    Ok(())
}

fn verify_sha256(path: &PathBuf, expected: &str) -> Result<bool> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open cached model file {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("failed to hash cached model file {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()) == expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_preference_defaults_to_auto() {
        assert_eq!(DevicePreference::default(), DevicePreference::Auto);
    }

    #[test]
    fn model_variants_use_the_names_expected_by_parakeet_rs() {
        for variant in [&FP16, &INT8] {
            let names = variant
                .assets
                .iter()
                .map(|asset| asset.local_name)
                .collect::<Vec<_>>();
            assert_eq!(
                names,
                [
                    "encoder-model.onnx",
                    "decoder_joint-model.onnx",
                    "vocab.txt"
                ]
            );
        }
    }
}
