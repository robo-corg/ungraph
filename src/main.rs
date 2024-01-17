


use std::io::stdout;
use std::{fs, fmt};
use std::path::{PathBuf, Path};

use clap::{Parser, ValueEnum};
use console::style;

use crate::model::Model;
use crate::onnx::OnnxModel;
use crate::safetensors::Safetensors;

mod safetensors;
mod onnx;
mod summary;
mod model;

#[derive(Debug, Copy, Clone, ValueEnum)]
enum OutputFormat {
    /// Text summary of model
    Text,
    /// Json summary of model
    Json
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputFormat::Text => write!(f, "text"),
            OutputFormat::Json => write!(f, "json"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model file to load
    model_file: PathBuf,
    #[arg(short, long, default_value_t=OutputFormat::Text)]
    output: OutputFormat
}

fn load_any_model(path: &Path) -> anyhow::Result<Box<dyn Model>> {
    let model_bytes = fs::read(path)?;

    if let Ok(onnx_model) = OnnxModel::from_bytes(model_bytes.as_slice()) {
        return Ok(Box::new(onnx_model));
    }

    let model = Safetensors::from_bytes(model_bytes.into())?;
    Ok(Box::new(model))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model = load_any_model(&args.model_file)?;
    let filename = args.model_file.file_name().and_then(|s| s.to_str());

    let summary = model.summary(filename);

    match args.output {
        OutputFormat::Text => {
            print!("{}", summary);
        },
        OutputFormat::Json => {
            let stdout = stdout();
            let mut stdout_lock = stdout.lock();
            summary.dump_json(&mut stdout_lock)?;
        },
    }

    Ok(())
}
