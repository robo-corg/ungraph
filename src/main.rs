


use std::io::stdout;
use std::{fs, fmt};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use console::style;

use crate::onnx::OnnxModel;

mod onnx;
mod summary;

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


fn main() {
    let args = Args::parse();

    let model_bytes = fs::read(&args.model_file).unwrap();

    let model = OnnxModel::from_bytes(model_bytes.as_slice()).unwrap();

    let summary = model.summary();

    match args.output {
        OutputFormat::Text => {
            print!("{}", summary);
        },
        OutputFormat::Json => {
            let stdout = stdout();
            let mut stdout_lock = stdout.lock();
            serde_json::to_writer_pretty(&mut stdout_lock, &summary).unwrap()
        },
    }

}
