use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

mod daemon;
mod model;
mod paths;

#[derive(Debug, Parser)]
#[command(name = "lw", about = "Experimental native Parakeet CUDA probe")]
struct Args {
    /// Directory containing the FP16 Parakeet TDT ONNX files.
    #[arg(long)]
    model_dir: PathBuf,

    /// A mono 16 kHz WAV file to transcribe.
    audio: PathBuf,
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("__daemon") {
        return daemon::serve();
    }

    let args = Args::parse();
    let mut model = model::Model::load(&args.model_dir)?;
    let text = model
        .transcribe(&args.audio)
        .with_context(|| format!("failed to transcribe {}", args.audio.display()))?;
    println!("{text}");

    Ok(())
}
