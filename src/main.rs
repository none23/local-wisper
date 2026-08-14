use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};

#[derive(Debug, Parser)]
#[command(name = "lw", about = "Experimental native Parakeet CUDA probe")]
struct Args {
    /// Directory containing the FP16 Parakeet TDT ONNX files.
    #[arg(long)]
    model_dir: PathBuf,

    /// A mono 16 kHz WAV file to transcribe.
    audio: PathBuf,
}

fn strict_cuda_config() -> ExecutionConfig {
    ExecutionConfig::new().with_custom_configure(|builder| {
        Ok(builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?)
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    let load_started = Instant::now();
    let mut model = ParakeetTDT::from_pretrained(&args.model_dir, Some(strict_cuda_config()))
        .with_context(|| {
            format!(
                "failed to load Parakeet with the CUDA execution provider from {}",
                args.model_dir.display()
            )
        })?;
    eprintln!("model loaded on CUDA in {:.2?}", load_started.elapsed());

    let inference_started = Instant::now();
    let result = model
        .transcribe_file(&args.audio, Some(TimestampMode::Sentences))
        .with_context(|| format!("failed to transcribe {}", args.audio.display()))?;
    eprintln!("transcribed in {:.2?}", inference_started.elapsed());
    println!("{}", result.text);

    Ok(())
}
