


use std::fs;
use std::path::PathBuf;

use clap::Parser;

use crate::onnx::OnnxModel;

mod onnx;
mod summary;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model file to load
    model_file: PathBuf,
}


fn main() {
    let args = Args::parse();

    let model_bytes = fs::read(&args.model_file).unwrap();

    let model = OnnxModel::from_bytes(model_bytes.as_slice()).unwrap();

    let summary = model.summary();

    print!("{}", summary);
}
