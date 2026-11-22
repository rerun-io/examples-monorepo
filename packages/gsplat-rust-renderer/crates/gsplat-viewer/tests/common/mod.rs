//! .rrd readers shared by the fixture tests.
use re_viewer::external::re_chunk::{Chunk, ComponentIdentifier};
use re_viewer::external::re_sdk_types::external::arrow::array::{
    Array as _, ArrayRef, AsArray as _, FixedSizeListArray,
};
use std::collections::BTreeMap;

/// Every chunk of an .rrd by entity path.
pub fn chunks(path: &str) -> BTreeMap<String, Vec<Chunk>> {
    let file = std::io::BufReader::new(std::fs::File::open(path).expect("open rrd"));
    let mut out: BTreeMap<String, Vec<Chunk>> = BTreeMap::new();
    for msg in re_log_encoding::DecoderApp::decode_eager(file).expect("rrd header") {
        if let re_log_msg::LogMsg::ArrowMsg(_, arrow) = msg.expect("rrd message") {
            let chunk = Chunk::from_arrow_msg(&arrow).expect("chunk");
            out.entry(chunk.entity_path().to_string())
                .or_default()
                .push(chunk);
        }
    }
    out
}

/// Row 0 of `archetype:field` (fixed-size lists flattened), or None when absent.
pub fn row(chunks: &[Chunk], archetype: &'static str, field: &str) -> Option<ArrayRef> {
    let id = ComponentIdentifier::from_archetype_field(archetype.into(), field);
    let array = chunks.iter().find_map(|c| c.raw_component_array(id))?;
    // A static chunk with one row: either a list column (take row 0) or the row's values directly.
    let row = match array.as_list_opt::<i32>() {
        Some(list) => list.value(0),
        None => array.clone(),
    };
    Some(match row.as_any().downcast_ref::<FixedSizeListArray>() {
        Some(fixed) => fixed.values().clone(),
        None => row,
    })
}
