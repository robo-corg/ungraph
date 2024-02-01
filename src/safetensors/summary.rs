use std::{fmt, io};

use serde::Serialize;

use crate::safetensors::Header;
use crate::summary::Summary;

#[derive(Serialize)]
pub struct SafeTensorsSummary<'a> {
    pub(crate) filename: Option<&'a str>,
    pub(crate) architecture: Option<&'a str>,
    pub(crate) implementation: Option<&'a str>,
    pub(crate) metadata: &'a Header,
    pub(crate) tensors: &'a Header,
}

impl<'a> Summary for SafeTensorsSummary<'a> {
    fn dump_json(&self, writer: &mut dyn io::Write) -> anyhow::Result<()> {
        Ok(serde_json::to_writer_pretty(writer, &self)?)
    }
}

impl<'a> fmt::Display for SafeTensorsSummary<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.filename.unwrap_or("<NO FILENAME>");
        writeln!(f, "Safetensors: {}", name)?;
        writeln!(f)?;

        if let Some(architecture) = self.architecture {
            writeln!(f, "architecture: {}", architecture)?;
        }

        if let Some(implementation) = self.implementation {
            writeln!(f, "implementation: {}", implementation)?;
        }

        Ok(())
    }
}
