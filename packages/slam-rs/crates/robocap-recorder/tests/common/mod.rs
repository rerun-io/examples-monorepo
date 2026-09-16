// Each integration test uses a different subset of these shared fixtures/helpers.
#![allow(dead_code)]
use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path};

use anyhow::{Context, Result};
use arrow_array::{Array, ArrayRef, ListArray};
use re_log_encoding::Decoder;
use re_log_types::LogMsg;

pub const IDR: [u8; 15] = [
    0, 0, 1, 0x67, 0x64, 0, 0, 1, 0x68, 0xef, 0, 0, 1, 0x65, 0x88,
];
pub const DELTA: [u8; 5] = [0, 0, 1, 0x41, 0x88];

type ComponentRows = BTreeMap<String, Vec<ArrayRef>>;

/// Component rows by entity and field, in recording order (without timeline columns).
pub fn rows(path: &Path) -> Result<BTreeMap<String, ComponentRows>> {
    let mut entities = BTreeMap::<String, ComponentRows>::new();
    for message in Decoder::<LogMsg>::decode_eager(BufReader::new(File::open(path)?))? {
        let LogMsg::ArrowMsg(_, message) = message? else {
            continue;
        };
        let schema = message.batch.schema();
        let entity = schema
            .metadata()
            .get("rerun:entity_path")
            .context("entity path missing")?;
        for (field, column) in schema.fields().iter().zip(message.batch.columns()) {
            let Some(list) = column.as_any().downcast_ref::<ListArray>() else {
                continue;
            };
            entities
                .entry(entity.clone())
                .or_default()
                .entry(field.name().clone())
                .or_default()
                .extend((0..list.len()).map(|row| list.value(row)));
        }
    }
    Ok(entities)
}

/// Typed values in one component row; a changed Arrow type fails the test.
pub fn values<T: Array + 'static>(row: &ArrayRef) -> Result<&T> {
    row.as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("unexpected component type: {:?}", row.data_type()))
}
