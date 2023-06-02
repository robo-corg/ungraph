use std::fmt;

use crate::onnx::ValueInfo;

pub struct OnnxOpset<'a> {
    pub name: &'a str,
    pub version: i64
}

pub struct OnnxSummary<'a> {
    pub domain: &'a str,
    pub name: &'a str,
    pub version: i64,
    pub doc_string: &'a str,
    pub producer_name: &'a str,
    pub producer_version: &'a str,
    pub ir_version: i64,
    pub opsets: Vec<OnnxOpset<'a>>,
    pub inputs: Vec<&'a ValueInfo>,
    pub outputs: Vec<&'a ValueInfo>,
    pub operator_summary: OperatorUsageSummary<'a>
}

pub struct OperatorUsage<'a> {
    pub name: &'a str,
    pub count: usize
}

pub struct OperatorUsageSummary<'a> {
    pub operators: Vec<OperatorUsage<'a>>
}

impl <'a> fmt::Display for OnnxSummary<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "ONNX Model: {} {} (v{})",
            self.domain,
            self.name,
            self.version
        )?;
        if self.doc_string.len() > 0 {
            writeln!(f, "{}", self.doc_string)?;
        }
        writeln!(f, "")?;

        writeln!(
            f,
            "Producer: {} {}",
            self.producer_name, self.producer_version
        )?;
        writeln!(f, "")?;

        writeln!(f, "IR Version: {}", self.ir_version)?;
        if self.opsets.len() == 1 {
            write!(f, "Opset: ")?;
        }

        for opset in self.opsets.iter() {
            writeln!(f, "{} {}", &opset.name, opset.version)?;
        }

        writeln!(f, "")?;
        writeln!(f, "Inputs:")?;

        for input in self.inputs.iter() {
            if input.source.is_none() {
                writeln!(f, "    {}: {}", input.name(), input.type_info())?;
            }
        }

        writeln!(f, "Outputs:")?;

        for output in self.outputs.iter() {
            writeln!(f, "    {}: {}", output.name(), output.type_info())?;
        }

        writeln

        Ok(())
    }
}