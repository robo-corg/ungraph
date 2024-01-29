use std::{fmt, io};

use console::Style;
use serde::{Serialize, Serializer};

use crate::onnx::TypeInfo;

pub trait Summary: fmt::Display {
    fn dump_json(&self, writer: &mut dyn io::Write) -> anyhow::Result<()> ;
}

#[derive(Serialize)]
pub struct OnnxOpset<'a> {
    pub name: &'a str,
    pub version: i64
}

#[derive(Serialize)]
pub struct Value<'a> {
    pub name: &'a str,
    #[serde(serialize_with = "type_info_serializer")]
    pub ty: TypeInfo<'a>
}

fn type_info_serializer<S>(ty: &TypeInfo<'_>, s: S) -> Result<S::Ok, S::Error> where S: Serializer {
    let ty_str = format!("{}", ty);
    s.serialize_str(&ty_str)
}

#[derive(Serialize)]
pub struct OnnxSummary<'a> {
    pub domain: &'a str,
    pub name: &'a str,
    pub version: i64,
    pub doc_string: &'a str,
    pub producer_name: &'a str,
    pub producer_version: &'a str,
    pub ir_version: i64,
    pub opsets: Vec<OnnxOpset<'a>>,
    pub inputs: Vec<Value<'a>>,
    pub outputs: Vec<Value<'a>>,
    pub operator_summary: OperatorUsageSummary<'a>
}

impl <'a> Summary for OnnxSummary<'a> {
    fn dump_json(&self, writer: &mut dyn io::Write) -> anyhow::Result<()> {
        Ok(serde_json::to_writer_pretty(writer, &self)?)
    }
}

#[derive(Serialize)]
pub struct OperatorUsage<'a> {
    pub domain: &'a str,
    pub name: &'a str,
    pub count: usize
}

#[derive(Serialize)]
pub struct OperatorUsageSummary<'a> {
    pub operators: Vec<OperatorUsage<'a>>
}

impl <'a> fmt::Display for OnnxSummary<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //println!("This is {} neat", style("quite").bold());
        let bold = Style::new().bold();
        writeln!(
            f,
            "{} {} {} (v{})",
            bold.apply_to("ONNX Model:"),
            self.domain,
            self.name,
            self.version
        )?;
        if !self.doc_string.is_empty() {
            writeln!(f, "{}", self.doc_string)?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "Producer: {} {}",
            self.producer_name, self.producer_version
        )?;
        writeln!(f)?;

        writeln!(f, "IR Version: {}", self.ir_version)?;
        if self.opsets.len() == 1 {
            write!(f, "Opset: ")?;
        }

        for opset in self.opsets.iter() {
            writeln!(f, "{} {}", &opset.name, opset.version)?;
        }

        writeln!(f)?;
        writeln!(f, "{}", bold.apply_to("Inputs:"))?;

        for input in self.inputs.iter() {
            writeln!(f, "    {}: {}", input.name, input.ty)?;
        }

        writeln!(f, "{}", bold.apply_to("Outputs:"))?;

        for output in self.outputs.iter() {
            writeln!(f, "    {}: {}", output.name, output.ty)?;
        }

        writeln!(f)?;
        writeln!(f, "Operators:")?;
        for oper in self.operator_summary.operators.iter() {
            writeln!(f, "    {}.{}: {}", oper.domain, oper.name, oper.count)?;
        }

        Ok(())
    }
}