use std::fmt;

use prost::bytes::Bytes;
use serde::Serialize;

type Header = serde_json::value::Map<String, serde_json::Value>;

pub struct Safetensors {
    header: Header,
    data: Bytes
}

impl Safetensors {
    pub fn from_bytes(data: Bytes) -> anyhow::Result<Self> {
        let header_size = u64::from_le_bytes(data[..8].try_into().unwrap());

        let header_end: usize = 8 + (header_size as usize);

        let header_bytes = data.slice(8..header_end);

        let header = serde_json::de::from_slice(&header_bytes)?;

        let data = data.slice(header_end..);

        Ok(Safetensors {
            header,
            data
        })
    }

    pub fn summary<'a>(&'a self) -> SafeTensorsSummary<'a> {
        SafeTensorsSummary {
            header: &self.header
        }
    }
}

#[derive(Serialize)]
pub struct SafeTensorsSummary<'a> {
    header: &'a Header
}

impl <'a> fmt::Display for SafeTensorsSummary<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Safetensor")
    }
}