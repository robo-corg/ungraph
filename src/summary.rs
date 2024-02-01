use std::{fmt, io};

pub trait Summary: fmt::Display {
    fn dump_json(&self, writer: &mut dyn io::Write) -> anyhow::Result<()>;
}
