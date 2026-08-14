use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

const REVISION: &str = "f88260fa0777fe0868dda6df85d1a98f012a4a7a";
const REPOSITORY: &str = "ysdede/parakeet-tdt-0.6b-v3-onnx";

struct Asset {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

const ASSETS: &[Asset] = &[
    Asset {
        name: "encoder-model.onnx",
        size: 1_238_960_452,
        sha256: "a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717",
    },
    Asset {
        name: "decoder_joint-model.onnx",
        size: 36_266_140,
        sha256: "b33a73b7c1d71b9d5a0911f5cb478be3dcbf79f53355c531ab1cd1dcd68ad8ef",
    },
    Asset {
        name: "vocab.txt",
        size: 102_132,
        sha256: "ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742",
    },
];

pub struct Model {
    inner: ParakeetTDT,
}

impl Model {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let started = Instant::now();
        let inner = ParakeetTDT::from_pretrained(model_dir, Some(strict_cuda_config()))
            .with_context(|| {
                format!(
                    "failed to load Parakeet with the CUDA execution provider from {}",
                    model_dir.display()
                )
            })?;
        eprintln!("model loaded on CUDA in {:.2?}", started.elapsed());
        Ok(Self { inner })
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

pub fn prepare(model_dir: &Path) -> Result<()> {
    fs::create_dir_all(model_dir)
        .with_context(|| format!("failed to create model cache {}", model_dir.display()))?;
    let marker = model_dir.join(".complete");
    if marker.is_file()
        && ASSETS
            .iter()
            .all(|asset| has_expected_size(model_dir, asset))
    {
        return Ok(());
    }

    let client = Client::builder()
        .build()
        .context("failed to initialize the model download client")?;
    for asset in ASSETS {
        let destination = model_dir.join(asset.name);
        if has_expected_size(model_dir, asset) && verify_sha256(&destination, asset.sha256)? {
            continue;
        }
        download_asset(&client, model_dir, asset)?;
    }

    let marker_part = model_dir.join(".complete.part");
    fs::write(&marker_part, format!("{REPOSITORY}@{REVISION}\n"))
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
    fs::metadata(model_dir.join(asset.name))
        .map(|metadata| metadata.len() == asset.size)
        .unwrap_or(false)
}

fn download_asset(client: &Client, model_dir: &Path, asset: &Asset) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{}",
        asset.name
    );
    eprintln!("downloading {}", asset.name);
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("failed to download {}", asset.name))?
        .error_for_status()
        .with_context(|| format!("model server rejected {}", asset.name))?;

    let part = model_dir.join(format!("{}.part", asset.name));
    let mut output = File::create(&part)
        .with_context(|| format!("failed to create partial model file {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut written = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = response
            .read(&mut buffer)
            .with_context(|| format!("failed while downloading {}", asset.name))?;
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
            asset.name,
            asset.size,
            asset.sha256,
            written,
            digest
        );
    }
    fs::rename(&part, model_dir.join(asset.name))
        .with_context(|| format!("failed to commit {} to the model cache", asset.name))?;
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
