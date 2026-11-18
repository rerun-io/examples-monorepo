//! Readers for the golden references of staged items: third-party outputs stored as `.rrd` fixtures on the NAS.
//!
//! The tests of this crate are `#[ignore]`d, because their assets live outside the repository. Run them with
//! `pixi run -e kornia-staging-dev --frozen kornia-staging-golden`, which skips, naming the asset, when it is absent.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _};
use arrow::array::{Array, ArrayRef, AsArray as _};
use arrow::datatypes::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int64Type, UInt32Type, UInt64Type,
};
use re_chunk::Chunk;
use re_log_msg::LogMsg;

/// Default location of the Basalt BRIEF / HashBoW golden on the NAS.
pub const BASALT_BRIEF_DEFAULT: &str =
    "/mnt/nas/ssd-8tb/scratch/slam-rs-rootslam/fixtures/b1/basalt_brief.rrd";

/// The Basalt BRIEF golden: `KORNIA_STAGING_BASALT_BRIEF`, else [`BASALT_BRIEF_DEFAULT`].
pub fn basalt_brief_path() -> PathBuf {
    std::env::var_os("KORNIA_STAGING_BASALT_BRIEF")
        .map_or_else(|| PathBuf::from(BASALT_BRIEF_DEFAULT), PathBuf::from)
}

/// One keypoint Basalt was given: its position, and its angle and descriptor when it was inside the 19-pixel border.
#[derive(Clone, Debug)]
pub struct BasaltKeypoint {
    /// `(x, y)` as the f32 Basalt received.
    pub position: [f32; 2],
    /// Basalt's `computeAngles` angle and `computeDescriptors` bits (`std::bitset<256>` words), `None` outside.
    pub described: Option<(f64, [u64; 4])>,
}

/// One input image of the harness.
#[derive(Clone, Debug)]
pub struct BasaltImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The keypoints, in the order Basalt received them.
    pub keypoints: Vec<BasaltKeypoint>,
}

/// Basalt's `matchDescriptors` result for one image pair: indices into each image's described keypoints.
#[derive(Clone, Debug)]
pub struct BasaltMatches {
    /// First image.
    pub image_a: usize,
    /// Second image.
    pub image_b: usize,
    /// `(index in a, index in b)`, sorted by the `a` index.
    pub pairs: Vec<(usize, usize)>,
}

/// `(frame, camera, score)` query hits, in Basalt's order.
pub type BasaltHits = Vec<(i64, u32, f64)>;

/// One `hash_bow.h` query of an indexed entry.
#[derive(Clone, Debug)]
pub struct BasaltQuery {
    /// The query entry's frame.
    pub frame: i64,
    /// The query entry's camera.
    pub cam: u32,
    /// Every entry that shares a word.
    pub all: BasaltHits,
    /// The top 10 among frames older than `frame - 5`.
    pub top: BasaltHits,
}

/// Output of `fixtures/basalt_brief/basalt_brief_dump.cpp` (Basalt's `keypoints.cpp` and `hash_bow.h`).
#[derive(Clone, Debug)]
pub struct BasaltBrief {
    /// The images, in harness order.
    pub images: Vec<BasaltImage>,
    /// One match set per image pair the harness matched.
    pub matches: Vec<BasaltMatches>,
    /// HashBoW bags `(word, weight)` by `(frame, camera)`.
    pub bags: BTreeMap<(i64, u32), Vec<(u32, f64)>>,
    /// The queries, in harness order.
    pub queries: Vec<BasaltQuery>,
}

/// The static components of each entity of a recording, by entity path and component name.
type Entities = BTreeMap<String, BTreeMap<String, ArrayRef>>;

fn read_entities(path: &Path) -> anyhow::Result<Entities> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut entities = Entities::new();
    for message in re_log_encoding::rrd::DecoderApp::decode_lazy(std::io::BufReader::new(file)) {
        let LogMsg::ArrowMsg(_, arrow_msg) = message? else {
            continue;
        };
        let chunk = Chunk::from_arrow_msg(&arrow_msg)?;
        if !chunk.is_static() {
            continue;
        }
        let components = entities.entry(chunk.entity_path().to_string()).or_default();
        for descriptor in chunk.component_descriptors() {
            if let Some(array) = chunk.raw_component_array(descriptor.component) {
                components.insert(descriptor.component.to_string(), array.clone());
            }
        }
    }
    Ok(entities)
}

/// The values of component `name`; empty when the entity does not carry it (the writer drops empty arrays).
fn values<T: ArrowPrimitiveType>(
    components: &BTreeMap<String, ArrayRef>,
    name: &str,
) -> anyhow::Result<Vec<T::Native>> {
    let Some(array) = components.get(name) else {
        return Ok(Vec::new());
    };
    let array = array
        .as_primitive_opt::<T>()
        .with_context(|| format!("{name}: type {}", array.data_type()))?;
    if array.null_count() > 0 {
        bail!("{name}: null values");
    }
    Ok(array.values().to_vec())
}

fn scalar<T: ArrowPrimitiveType>(
    components: &BTreeMap<String, ArrayRef>,
    name: &str,
) -> anyhow::Result<T::Native> {
    match values::<T>(components, name)?.as_slice() {
        [value] => Ok(*value),
        other => bail!("{name}: expected one value, got {}", other.len()),
    }
}

/// The entities under `prefix/<index>`, in index order.
fn indexed<'a>(
    entities: &'a Entities,
    prefix: &str,
) -> anyhow::Result<Vec<&'a BTreeMap<String, ArrayRef>>> {
    let mut found: Vec<(usize, &BTreeMap<String, ArrayRef>)> = entities
        .iter()
        .filter_map(|(path, components)| {
            Some((path.strip_prefix(prefix)?.parse().ok()?, components))
        })
        .collect();
    found.sort_by_key(|&(index, _)| index);
    if found.iter().enumerate().any(|(i, &(index, _))| i != index) {
        bail!("{prefix}<index>: indices are not 0..{}", found.len());
    }
    Ok(found
        .into_iter()
        .map(|(_, components)| components)
        .collect())
}

fn hits(components: &BTreeMap<String, ArrayRef>, kind: &str) -> anyhow::Result<BasaltHits> {
    let frames = values::<Int64Type>(components, &format!("{kind}_frame"))?;
    let cams = values::<UInt32Type>(components, &format!("{kind}_cam"))?;
    let scores = values::<Float64Type>(components, &format!("{kind}_score"))?;
    if frames.len() != cams.len() || frames.len() != scores.len() {
        bail!("{kind}: column lengths differ");
    }
    Ok(frames
        .into_iter()
        .zip(cams)
        .zip(scores)
        .map(|((f, c), s)| (f, c, s))
        .collect())
}

impl BasaltBrief {
    /// Read the golden `.rrd` written by `fixtures/basalt_brief/basalt_brief_to_rrd.py`.
    ///
    /// # Errors
    ///
    /// When the file is missing, is not an `.rrd`, or its entities do not have the documented layout.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let entities = read_entities(path)?;
        let images = indexed(&entities, "/images/")?
            .into_iter()
            .map(|c| {
                let x = values::<Float32Type>(c, "x")?;
                let y = values::<Float32Type>(c, "y")?;
                let described = c
                    .get("described")
                    .and_then(|a| a.as_boolean_opt())
                    .context("described: missing or not boolean")?;
                let angle = values::<Float64Type>(c, "angle")?;
                let descriptor = values::<UInt64Type>(c, "descriptor")?;
                let n = x.len();
                if y.len() != n
                    || described.len() != n
                    || angle.len() != n
                    || descriptor.len() != 4 * n
                {
                    bail!("image keypoint columns differ in length");
                }
                let keypoints = (0..n)
                    .map(|i| BasaltKeypoint {
                        position: [x[i], y[i]],
                        described: described.value(i).then(|| {
                            (
                                angle[i],
                                [
                                    descriptor[4 * i],
                                    descriptor[4 * i + 1],
                                    descriptor[4 * i + 2],
                                    descriptor[4 * i + 3],
                                ],
                            )
                        }),
                    })
                    .collect();
                Ok(BasaltImage {
                    width: scalar::<UInt32Type>(c, "width")? as usize,
                    height: scalar::<UInt32Type>(c, "height")? as usize,
                    keypoints,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        let matches = indexed(&entities, "/matches/")?
            .into_iter()
            .map(|c| {
                let a = values::<UInt32Type>(c, "index_a")?;
                let b = values::<UInt32Type>(c, "index_b")?;
                if a.len() != b.len() {
                    bail!("match columns differ in length");
                }
                Ok(BasaltMatches {
                    image_a: scalar::<UInt32Type>(c, "image_a")? as usize,
                    image_b: scalar::<UInt32Type>(c, "image_b")? as usize,
                    pairs: a
                        .into_iter()
                        .zip(b)
                        .map(|(a, b)| (a as usize, b as usize))
                        .collect(),
                })
            })
            .collect::<anyhow::Result<_>>()?;
        let bags = indexed(&entities, "/bags/")?
            .into_iter()
            .map(|c| {
                let words = values::<UInt32Type>(c, "word")?;
                let weights = values::<Float64Type>(c, "weight")?;
                if words.len() != weights.len() {
                    bail!("bag columns differ in length");
                }
                let key = (
                    scalar::<Int64Type>(c, "frame")?,
                    scalar::<UInt32Type>(c, "cam")?,
                );
                Ok((key, words.into_iter().zip(weights).collect()))
            })
            .collect::<anyhow::Result<_>>()?;
        let queries = indexed(&entities, "/queries/")?
            .into_iter()
            .map(|c| {
                Ok(BasaltQuery {
                    frame: scalar::<Int64Type>(c, "frame")?,
                    cam: scalar::<UInt32Type>(c, "cam")?,
                    all: hits(c, "all")?,
                    top: hits(c, "top")?,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            images,
            matches,
            bags,
            queries,
        })
    }

    /// [`BasaltBrief::load`] of [`basalt_brief_path`].
    ///
    /// # Panics
    ///
    /// When the golden is absent or unreadable; the message names the asset.
    pub fn load_default() -> Self {
        let path = basalt_brief_path();
        Self::load(&path).unwrap_or_else(|e| {
            panic!(
                "needs the Basalt BRIEF golden {} (set KORNIA_STAGING_BASALT_BRIEF): {e:#}",
                path.display()
            )
        })
    }
}
