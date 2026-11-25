//! The set on the GPU: every loaded file de-quantised at its row offset into one scene, and the index measured from
//! the decoded scales (feature size per level and per leaf, splat reach per leaf).
use super::{Error, Leaf, LodIndex, LodMeta, Run};
use glam::Vec3;

/// One SOG file of a set, as bytes (the core does no file I/O).
#[derive(Clone, Copy, Debug)]
pub struct LodFile<'a> {
    pub meta_json: &'a str,
    pub webp: crate::sog::SogWebp<'a>,
}

/// A Streamed SOG set: the index and its files in `meta.filenames` order (`None` = not available, skipped).
#[derive(Clone, Debug)]
pub struct LodSet<'a> {
    pub meta: &'a LodMeta,
    pub files: Vec<Option<LodFile<'a>>>,
}

/// Load-time choices.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Drop levels finer than this (0 = keep all), for scenes that do not fit the device.
    pub min_level: u32,
    /// Percentile (0..1) of a leaf's splat sizes at a level that rates the leaf at that level (default 0.9).
    pub feature_percentile: f32,
    /// Clamp opacity to 1 as single SOGs do (merged v3 levels store up to 1.5625); default keeps it linear.
    pub clamp_opacity: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            min_level: 0,
            feature_percentile: 0.9,
            clamp_opacity: false,
        }
    }
}

/// Per row `exp(max log scale)` (1 sigma of the major axis), from the decoded scales plane and its codebook.
fn push_sizes(sizes: &mut Vec<f32>, q: &crate::sog::Quantization, scales: &[u8]) {
    sizes.extend(scales.as_chunks::<4>().0.iter().take(q.count).map(|t| {
        q.scales[t[0] as usize]
            .max(q.scales[t[1] as usize])
            .max(q.scales[t[2] as usize])
            .exp()
    }));
}

/// The `p` quantile (0..1, nearest rank) of `values`; NaN when empty.
fn quantile(mut values: impl AsMut<[f32]>, p: f32) -> f32 {
    let values = values.as_mut();
    if values.is_empty() {
        return f32::NAN;
    }
    let k = ((values.len() - 1) as f32 * p.clamp(0.0, 1.0)).round() as usize;
    *values.select_nth_unstable_by(k, f32::total_cmp).1
}

/// The finest level >= `requested` whose files (per file: level, rows, SH degree bound) fit `fits(rows, degree)`;
/// `None` when not even the coarsest level fits (or there are no files).
pub fn fit_min_level(
    files: &[(u32, usize, u32)],
    requested: u32,
    fits: impl Fn(usize, u32) -> bool,
) -> Option<u32> {
    let coarsest = files.iter().map(|f| f.0).max()?;
    (requested..=coarsest).find(|&m| {
        let kept = || files.iter().filter(|f| f.0 >= m);
        fits(
            kept().map(|f| f.1).sum(),
            kept().map(|f| f.2).max().unwrap_or(0),
        )
    })
}

/// The minimum level [`upload`] loads under `limit`: [`fit_min_level`], or, when not even the coarsest level fits,
/// that level's capacity error (before any file is decoded). With no file the requested level stands.
fn loaded_min_level(
    files: &[(u32, usize, u32)],
    requested: u32,
    limit: u64,
) -> Result<u32, crate::Error> {
    let bytes = |rows, degree| crate::Splats::gpu_bytes(rows, degree);
    if let Some(level) = fit_min_level(files, requested, |rows, degree| {
        bytes(rows, degree) <= limit
    }) {
        return Ok(level);
    }
    let Some(coarsest) = files.iter().map(|f| f.0).max() else {
        return Ok(requested);
    };
    let kept = || files.iter().filter(|f| f.0 == coarsest);
    Err(crate::Error::Capacity {
        required: bytes(
            kept().map(|f| f.1).sum(),
            kept().map(|f| f.2).max().unwrap_or(0),
        ),
        limit,
    })
}

/// Decode every loaded file (CPU WebP), de-quantise each on the GPU at its row offset into one scene, and build the
/// index. Files are placed in (level, file) order; levels finer than `options.min_level` and `None` files are skipped.
/// When those files do not fit the device's binding limit, finer levels are dropped until they do (the index's
/// [`LodIndex::min_level`] then says which level is the finest loaded).
pub(crate) fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &crate::gpu::Kernel,
    set: &LodSet<'_>,
    options: &LoadOptions,
    limit: u64,
) -> Result<(crate::Scene, LodIndex), crate::Error> {
    let meta = set.meta;
    if set.files.len() != meta.filenames.len() {
        return Err(Error::Invalid("one file slot per filename".into()).into());
    }
    let mut order: Vec<(u32, usize)> = meta
        .loaded_files(options.min_level)?
        .into_iter()
        .filter(|&(f, _)| set.files[f].is_some())
        .map(|(f, level)| (level, f))
        .collect();
    order.sort_unstable();
    let mut metas = Vec::with_capacity(order.len());
    for &(level, f) in &order {
        let file = set.files[f].as_ref().expect("filtered above");
        metas.push((level, f, crate::sog::Meta::parse(file.meta_json)?));
    }
    let sizes: Vec<(u32, usize, u32)> = metas
        .iter()
        .map(|(level, _, m)| (*level, m.count, m.max_sh_degree()))
        .collect();
    let fitted = loaded_min_level(&sizes, options.min_level, limit)?;
    let mut decoded = Vec::with_capacity(metas.len());
    for (level, f, sog_meta) in metas.into_iter().filter(|m| m.0 >= fitted) {
        let file = set.files[f].as_ref().expect("filtered above");
        decoded.push((level, f, crate::sog::decode_planes(&sog_meta, &file.webp)?));
    }
    let rest = decoded
        .iter()
        .filter_map(|(_, _, d)| {
            d.sh.as_ref()
                .map(|s| crate::sog::rest_coefficients(s.bands))
        })
        .max()
        .unwrap_or(0);
    let total: usize = decoded.iter().map(|(_, _, d)| d.quantization.count).sum();
    let n = crate::Scene::count(total)?;
    let mut out = crate::sog_gpu::Outputs::new(device, total, rest, limit)?;
    out.linear_opacity = !options.clamp_opacity;
    let mut file_first = vec![None; meta.filenames.len()];
    let mut file_count = vec![0u32; meta.filenames.len()];
    let mut ranges = Vec::new();
    let mut sizes: Vec<f32> = Vec::with_capacity(total);
    let mut next = 0u32;
    for (level, f, d) in &decoded {
        crate::sog_gpu::dequantize_into(
            device,
            queue,
            kernel,
            &d.quantization,
            &d.planes(),
            limit,
            &out,
            next,
        )?;
        let count = d.quantization.count as u32;
        file_first[*f] = Some(next);
        file_count[*f] = count;
        ranges.push((next, next + count, *level));
        push_sizes(&mut sizes, &d.quantization, &d.scales.data);
        next += count;
    }
    drop(decoded);
    // One scratch buffer for every quantile; the files of a level are one contiguous span of rows.
    let mut scratch = Vec::new();
    let mut quantile_of = |rows: &[f32], p: f32| {
        scratch.clear();
        scratch.extend_from_slice(rows);
        quantile(&mut scratch[..], p)
    };
    let feature = (0..meta.levels)
        .map(|level| {
            let span = ranges
                .iter()
                .filter(|r| r.2 == level)
                .map(|r| (r.0, r.1))
                .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
                .unwrap_or((0, 0));
            quantile_of(&sizes[span.0 as usize..span.1 as usize], 0.5)
        })
        .collect();
    let mut leaves = Vec::with_capacity(meta.leaves.len());
    for leaf in &meta.leaves {
        let mut runs = Vec::with_capacity(leaf.lods.len());
        for run in &leaf.lods {
            runs.push(match run {
                Some(r) => match file_first[r.file as usize] {
                    Some(first) => {
                        if u64::from(r.offset) + u64::from(r.count)
                            > u64::from(file_count[r.file as usize])
                        {
                            return Err(Error::Invalid(format!(
                                "leaf run {r:?} exceeds the {} rows of its file",
                                file_count[r.file as usize]
                            ))
                            .into());
                        }
                        Some(Run {
                            first: first + r.offset,
                            count: r.count,
                        })
                    }
                    None => None,
                },
                None => None,
            });
        }
        let feature = runs
            .iter()
            .map(|run| {
                run.map_or(f32::NAN, |r| {
                    quantile_of(
                        &sizes[r.first as usize..r.end() as usize],
                        options.feature_percentile,
                    )
                })
            })
            .collect();
        let reach = runs
            .iter()
            .flatten()
            .flat_map(|r| sizes[r.first as usize..r.end() as usize].iter().copied())
            .fold(0.0f32, f32::max)
            * 3.0;
        leaves.push(Leaf {
            min: Vec3::from(leaf.min),
            max: Vec3::from(leaf.max),
            runs,
            feature,
            reach,
        });
    }
    let index = LodIndex::new(leaves, feature, ranges)?;
    Ok((crate::sog_gpu::scene(device, out, n), index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_min_level_drops_finer_levels_until_the_set_fits() {
        // Levels 0, 1, 2 with 100, 50, 25 rows; level 0 has SH 3, the others none.
        let files = [(0, 100, 3), (1, 50, 0), (2, 25, 0), (1, 10, 0)];
        let fits =
            |limit: usize| move |rows: usize, degree: u32| rows * (degree as usize + 1) <= limit;
        assert_eq!(fit_min_level(&files, 0, fits(10_000)), Some(0));
        assert_eq!(fit_min_level(&files, 0, fits(500)), Some(1));
        assert_eq!(fit_min_level(&files, 2, fits(10_000)), Some(2));
        assert_eq!(fit_min_level(&files, 0, fits(30)), Some(2));
        assert_eq!(fit_min_level(&files, 0, fits(10)), None);
        assert_eq!(fit_min_level(&[], 0, fits(10)), None);
    }

    #[test]
    fn loaded_min_level_refuses_a_set_whose_coarsest_level_does_not_fit() {
        // Level 0: 100 rows with SH 3 (1600 bytes each); level 1: 50 + 10 rows, no SH (40 bytes each).
        let files = [(0, 100, 3), (1, 50, 0), (1, 10, 0)];
        assert_eq!(loaded_min_level(&files, 0, u64::MAX).unwrap(), 0);
        assert_eq!(loaded_min_level(&files, 0, 60 * 40).unwrap(), 1);
        assert!(matches!(
            loaded_min_level(&files, 0, 60 * 40 - 1),
            Err(crate::Error::Capacity {
                required: 2400,
                limit: 2399
            })
        ));
        // A set with no loaded file keeps the requested level (an empty scene).
        assert_eq!(loaded_min_level(&[], 2, 0).unwrap(), 2);
    }

    #[test]
    fn quantile_takes_the_nearest_rank() {
        assert_eq!(
            quantile((0..11).map(|i| i as f32).collect::<Vec<_>>(), 0.9),
            9.0
        );
        assert_eq!(quantile(vec![3.0], 0.5), 3.0);
        assert!(quantile(Vec::<f32>::new(), 0.5).is_nan());
    }

    #[test]
    fn linear_opacity_keeps_merged_v3_opacity_above_one() {
        use crate::sog::Version;
        assert_eq!(crate::sog_gpu::linear_opacity(Version::V3, 255), 1.5625);
        assert_eq!(crate::sog_gpu::linear_opacity(Version::V3, 204), 1.0);
        assert_eq!(crate::sog_gpu::linear_opacity(Version::V2, 255), 1.0);
        let q = crate::sog::Quantization {
            version: Version::V3,
            count: 0,
            means_min: [0.0; 3],
            means_max: [1.0; 3],
            scales: [0.0; 256],
            sh0: [0.0; 256],
            sh_rest: [0.0; 256],
        };
        assert_eq!(crate::sog_gpu::tables_with(&q, true)[768 + 255], 1.5625);
        assert!(crate::sog_gpu::tables_with(&q, false)[768 + 255] < 14.0);
    }
}
