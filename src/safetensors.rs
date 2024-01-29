use std::{fmt, io};

use anyhow::bail;
use prost::bytes::Bytes;
use serde::Serialize;
use serde_json::from_value;

use crate::model::Model;
use crate::summary::Summary;

type Header = serde_json::value::Map<String, serde_json::Value>;

pub struct Safetensors {
    metadata: Header,
    tensors: Header,
}

impl Safetensors {
    pub fn from_bytes(data: Bytes) -> anyhow::Result<Self> {
        let header_size = u64::from_le_bytes(data[..8].try_into().unwrap());

        let header_end: usize = 8 + (header_size as usize);

        if header_end > data.len() {
            bail!("Header larger remaining file length");
        }

        let header_bytes = data.slice(8..header_end);

        let mut header: Header = serde_json::de::from_slice(&header_bytes)?;

        let _data = data.slice(header_end..);

        let metadata = header
            .remove("__metadata__")
            .and_then(|v| from_value(v).ok())
            .unwrap_or_default();

        let tensors = header;

        Ok(Safetensors { metadata, tensors })
    }
}

impl Model for Safetensors {
    fn summary<'a>(&'a self, filename: Option<&'a str>) -> Box<dyn Summary + 'a> {
        let architecture = self
            .metadata
            .get("modelspec.architecture")
            .and_then(|v| v.as_str());
        let implementation = self
            .metadata
            .get("modelspec.implementation")
            .and_then(|v| v.as_str());

        // let data_types =

        Box::new(SafeTensorsSummary {
            filename,
            architecture,
            implementation,
            metadata: &self.metadata,
            tensors: &self.tensors,
        })
    }
}

#[derive(Serialize)]
pub struct SafeTensorsSummary<'a> {
    filename: Option<&'a str>,
    architecture: Option<&'a str>,
    implementation: Option<&'a str>,
    metadata: &'a Header,
    tensors: &'a Header,
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
