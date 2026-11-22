//! `gsplat_core::sog` against the numpy oracle (`gsplat_rust_renderer.sog_decode`, a port of splat-transform's
//! decode): `tests/fixtures/sog_synthetic.rrd` holds a v2 and a v3 SOG as `SogSplats3D` (written by
//! `tests/test_sog_decode.py`) and the oracle decode under `/expected/<name>`. Set `GSPLAT_SOG_ORACLE_RRD` to check
//! another .rrd of the same layout (the skull.sog integration test in `tests/test_sog_decode.py` does).
use gsplat_core::sog::{SogWebp, decode};
use re_viewer::external::re_chunk::{Chunk, ComponentIdentifier};
use re_viewer::external::re_sdk_types::external::arrow::array::{
    Array as _, ArrayRef, AsArray as _, FixedSizeListArray,
};
use re_viewer::external::re_sdk_types::external::arrow::datatypes::{Float32Type, UInt8Type};
use std::collections::BTreeMap;

/// Every chunk of an .rrd by entity path.
fn chunks(path: &str) -> BTreeMap<String, Vec<Chunk>> {
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
fn row(chunks: &[Chunk], archetype: &'static str, field: &str) -> Option<ArrayRef> {
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

fn bytes(chunks: &[Chunk], field: &str) -> Option<Vec<u8>> {
    row(chunks, "SogSplats3D", field).map(|a| a.as_primitive::<UInt8Type>().values().to_vec())
}

fn expected(chunks: &[Chunk], field: &str) -> Vec<f32> {
    row(chunks, "Expected", field)
        .unwrap_or_else(|| panic!("Expected:{field} missing"))
        .as_primitive::<Float32Type>()
        .values()
        .to_vec()
}

/// Relative difference with a floor of 1 on the scale (absolute below 1).
fn rel(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs().max(1.0)
}

/// Rust decode == oracle (<= 1e-6 relative) for every SOG entity of `path`; returns how many were checked.
fn check(path: &str) -> usize {
    let all = chunks(path);
    let mut checked = 0;
    for (entity, entity_chunks) in &all {
        let Some(meta) = row(entity_chunks, "SogSplats3D", "meta") else {
            continue;
        };
        let meta = meta.as_string::<i32>().value(0).to_owned();
        let format = row(entity_chunks, "SogSplats3D", "format").expect("format");
        let planes: BTreeMap<&str, Vec<u8>> = [
            "means_l",
            "means_u",
            "quats",
            "scales",
            "sh0",
            "shN_centroids",
            "shN_labels",
        ]
        .into_iter()
        .filter_map(|p| bytes(entity_chunks, p).map(|b| (p, b)))
        .collect();
        let webp = SogWebp {
            means_l: &planes["means_l"],
            means_u: &planes["means_u"],
            quats: &planes["quats"],
            scales: &planes["scales"],
            sh0: &planes["sh0"],
            sh_centroids: planes.get("shN_centroids").map(Vec::as_slice),
            sh_labels: planes.get("shN_labels").map(Vec::as_slice),
        };
        let start = std::time::Instant::now();
        let splats = decode(&meta, &webp).expect("decode");
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let reference = &all[&format!("/expected{entity}")];
        let transforms = expected(reference, "transforms");
        let opacities = expected(reference, "raw_opacities");
        let sh = expected(reference, "sh");
        let n = opacities.len();
        assert_eq!(splats.transforms.len(), n, "{entity}: splat count");
        assert_eq!(
            splats.sh_coefficients.len() * 3,
            sh.len(),
            "{entity}: SH size"
        );
        let mut worst = [0.0f32; 3];
        for (i, t) in splats.transforms.iter().enumerate() {
            for (k, v) in t.iter().enumerate() {
                worst[0] = worst[0].max(rel(*v, transforms[i * 10 + k]));
            }
            worst[1] = worst[1].max(rel(splats.raw_opacities[i], opacities[i]));
        }
        for (i, c) in splats.sh_coefficients.iter().flatten().enumerate() {
            worst[2] = worst[2].max(rel(*c, sh[i]));
        }
        println!(
            "{entity} ({}): {n} splats, SH degree {}, decode {ms:.1} ms, worst rel transforms {:.2e} logit {:.2e} sh {:.2e}",
            format.as_string::<i32>().value(0),
            splats.sh_degree,
            worst[0],
            worst[1],
            worst[2]
        );
        assert!(worst.iter().all(|w| *w <= 1e-6), "{entity}: {worst:?}");
        checked += 1;
    }
    checked
}

#[test]
fn synthetic_sog_matches_oracle() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sog_synthetic.rrd"
    );
    assert_eq!(check(path), 2);
}

#[test]
fn oracle_rrd_from_env_matches() {
    let Ok(path) = std::env::var("GSPLAT_SOG_ORACLE_RRD") else {
        println!("GSPLAT_SOG_ORACLE_RRD not set: nothing to check");
        return;
    };
    assert!(check(&path) > 0, "{path}: no SogSplats3D entity");
}
