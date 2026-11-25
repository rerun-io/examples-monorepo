//! `SogLodSplats3D`: a `PlayCanvas` Streamed SOG (lod-meta.json + one SOG per file) logged as stored, on the same
//! entity as a static `GaussianSplats3D` preview (what stock viewers draw). Writer: `gsplat_rust_renderer/sog3d.py`.
//!
//! Every file of `filenames` is decoded once per payload (WebP on the CPU, de-quantisation on the GPU) into ONE
//! shared scene with its [`gsplat_core::lod::LodIndex`]. Each view then selects its rows on the host
//! ([`gsplat_core::lod::select`]) when it encodes a new frame, so a still camera selects and encodes nothing.
//! The environment file (if any) is not drawn (the core does not load it yet).
//!
//! Parameters: each view's selection reads [`gsplat_core::lod::LodParams`] from its `RenderOptions::lod` (part of
//! the view's image and selection key). Their defaults are [`LodParams::default`] (the same as the CLI), which these
//! environment variables override once per process (no blueprint setting yet):
//! `GSPLAT_LOD_MODE` = `off` | `cull` | `lod`, `GSPLAT_LOD_PX` = tau in pixels, `GSPLAT_LOD_BUDGET` = splat budget in
//! rows, `GSPLAT_LOD_MIN_LEVEL` = finest level loaded and drawn.
use gsplat_core::lod::{LodMeta, LodParams};
use gsplat_core::native::NativeSplats;
use gsplat_core::sog::SogWebp;
use re_viewer::external::{re_log, re_renderer, re_sdk_types, re_viewer_context};
use std::sync::{Arc, OnceLock};

pub(crate) const ARCHETYPE: &str = "SogLodSplats3D";

/// Plane fields in the writer's order (`plane_sizes` is file-major over these).
pub(crate) const PLANES: [&str; 7] = [
    "means_l",
    "means_u",
    "quats",
    "scales",
    "sh0",
    "shN_centroids",
    "shN_labels",
];

/// lod-meta.json, the per-file meta.json texts, the plane sizes, then the planes.
pub(crate) const FIELDS: [&str; 10] = [
    "lod_meta",
    "metas",
    "plane_sizes",
    PLANES[0],
    PLANES[1],
    PLANES[2],
    PLANES[3],
    PLANES[4],
    PLANES[5],
    PLANES[6],
];

pub(crate) fn components() -> [re_sdk_types::ComponentIdentifier; 10] {
    FIELDS.map(|field| {
        re_sdk_types::ComponentIdentifier::from_archetype_field(ARCHETYPE.into(), field)
    })
}

/// What the visualizer needs every frame, parsed once per payload generation.
#[derive(Debug)]
pub(crate) struct Info {
    pub meta: LodMeta,
    /// Rows of the files the scene will hold (levels >= the load-time minimum level).
    pub count: usize,
    /// Upper bound of the SH degree over those files.
    pub max_degree: u32,
    /// Per `filenames` entry: whether it is loaded (its level >= the minimum level).
    pub loaded: Vec<bool>,
    /// Per loaded file: (level, rows, SH degree bound), for the device fit of the minimum level.
    pub sizes: Vec<(u32, usize, u32)>,
}

fn info(lod_meta: &str, metas: &[&str]) -> Result<Info, String> {
    let meta = LodMeta::parse(lod_meta).map_err(|e| e.to_string())?;
    if metas.len() < meta.filenames.len() {
        return Err(format!(
            "SogLodSplats3D: {} meta.json texts for {} files",
            metas.len(),
            meta.filenames.len()
        ));
    }
    let mut count = 0;
    let mut max_degree = 0;
    let mut loaded = vec![false; meta.filenames.len()];
    let mut sizes = Vec::new();
    for (file, level) in meta
        .loaded_files(params().min_level)
        .map_err(|e| e.to_string())?
    {
        loaded[file] = true;
        let file = gsplat_core::sog::Meta::parse(metas[file]).map_err(|e| e.to_string())?;
        let degree = file.max_sh_degree();
        count += file.count;
        max_degree = max_degree.max(degree);
        sizes.push((level, file.count, degree));
    }
    Ok(Info {
        meta,
        count,
        max_degree,
        loaded,
        sizes,
    })
}

/// One entity's Streamed SOG payload plus the preview rows it replaces.
pub(crate) struct Payload<'a> {
    pub info: &'a Info,
    pub metas: &'a [&'a str],
    /// Per `filenames` entry (the environment, logged last, is not included).
    pub webp: &'a [SogWebp<'a>],
    /// The same entity's `GaussianSplats3D` rows: bounds, and the native fallback.
    pub preview: &'a NativeSplats<'a>,
}

impl Payload<'_> {
    pub(crate) fn count(&self) -> usize {
        self.info.count
    }

    /// Fits when some minimum level fits the device (the upload drops finer levels until one does).
    pub(crate) fn check_capacity(
        &self,
        core: &gsplat_core::Renderer,
    ) -> Result<(), gsplat_core::Error> {
        match core.fit_lod_min_level(&self.info.sizes, params().min_level) {
            Some(_) => Ok(()),
            None => core.check_capacity(self.info.count, self.info.max_degree),
        }
    }

    /// All loaded files into one scene (CPU WebP, GPU de-quantisation per file) plus the host-side index.
    pub(crate) fn upload(
        &self,
        ctx: &re_renderer::RenderContext,
        core: &gsplat_core::Renderer,
    ) -> Result<(Arc<gsplat_core::Scene>, Arc<gsplat_core::lod::LodIndex>), gsplat_core::Error>
    {
        re_tracing::profile_function!();
        let start = web_time::Instant::now();
        let set = gsplat_core::lod::LodSet {
            meta: &self.info.meta,
            files: (0..self.info.meta.filenames.len())
                .map(|i| {
                    self.info.loaded[i].then(|| gsplat_core::lod::LodFile {
                        meta_json: self.metas[i],
                        webp: self.webp[i],
                    })
                })
                .collect(),
        };
        // Merged SOG v3 levels keep their opacity above 1 (the core's default).
        let options = gsplat_core::lod::LoadOptions {
            min_level: params().min_level,
            ..Default::default()
        };
        let (scene, index) = core.upload_lod(&ctx.queue, &set, &options)?;
        #[cfg(not(target_arch = "wasm32"))]
        ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
        re_log::info!(
            "Streamed SOG decode: {} rows in {} levels (finest loaded {}, requested {}; {} files, {} leaves), WebP + upload + de-quantise {:.1} ms",
            re_format::format_uint(scene.len()),
            index.levels(),
            index.min_level(),
            options.min_level,
            // Files the upload kept (the device fit may drop finer levels).
            self.info
                .sizes
                .iter()
                .filter(|(level, ..)| *level >= index.min_level())
                .count(),
            index.leaves().len(),
            start.elapsed().as_secs_f64() * 1e3,
        );
        Ok((scene, index))
    }
}

/// Split the concatenated planes into per-file WebP slices (`plane_sizes` is file-major, 0 = absent).
pub(crate) fn split_planes<'a>(
    files: usize,
    sizes: &[u32],
    planes: [Option<&'a [u8]>; 7],
) -> Result<Vec<SogWebp<'a>>, String> {
    if !sizes.len().is_multiple_of(PLANES.len()) || sizes.len() / PLANES.len() < files {
        return Err(format!(
            "SogLodSplats3D: {} plane sizes for {files} files",
            sizes.len()
        ));
    }
    let mut offsets = [0usize; 7];
    let mut out = Vec::with_capacity(files);
    for file in 0..files {
        let mut slices: [Option<&'a [u8]>; 7] = [None; 7];
        for (p, slice) in slices.iter_mut().enumerate() {
            let size = sizes[file * PLANES.len() + p] as usize;
            if size == 0 {
                continue;
            }
            let plane = planes[p]
                .ok_or_else(|| format!("SogLodSplats3D: plane {} is missing", PLANES[p]))?;
            let range = offsets[p]..offsets[p] + size;
            *slice = Some(plane.get(range).ok_or_else(|| {
                format!(
                    "SogLodSplats3D: plane {} is shorter than its sizes",
                    PLANES[p]
                )
            })?);
            offsets[p] += size;
        }
        let required = |p: usize| {
            slices[p]
                .ok_or_else(|| format!("SogLodSplats3D: file {file} has no {} plane", PLANES[p]))
        };
        out.push(SogWebp {
            means_l: required(0)?,
            means_u: required(1)?,
            quats: required(2)?,
            scales: required(3)?,
            sh0: required(4)?,
            sh_centroids: slices[5],
            sh_labels: slices[6],
        });
    }
    Ok(out)
}

/// Default selection parameters ([`LodParams::default`] seeded by the environment), read once per process.
pub(crate) fn params() -> LodParams {
    static PARAMS: OnceLock<LodParams> = OnceLock::new();
    *PARAMS.get_or_init(|| {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        let params = parse_params(
            var("GSPLAT_LOD_MODE").as_deref(),
            var("GSPLAT_LOD_PX").as_deref(),
            var("GSPLAT_LOD_BUDGET").as_deref(),
            var("GSPLAT_LOD_MIN_LEVEL").as_deref(),
        )
        .unwrap_or_else(|error| {
            re_log::warn!("{error}; using the default LOD parameters");
            LodParams::default()
        });
        re_log::info!("Streamed SOG LOD parameters: {params:?}");
        params
    })
}

pub(crate) fn parse_params(
    mode: Option<&str>,
    tau_px: Option<&str>,
    budget: Option<&str>,
    min_level: Option<&str>,
) -> Result<LodParams, String> {
    let default = LodParams::default();
    Ok(LodParams {
        mode: match mode {
            None => default.mode,
            Some(v) => v
                .parse()
                .map_err(|e: String| format!("GSPLAT_LOD_MODE: {e}"))?,
        },
        tau_px: match tau_px {
            None => default.tau_px,
            Some(v) => v
                .parse::<f32>()
                .ok()
                .filter(|t| t.is_finite() && *t > 0.0)
                .ok_or_else(|| format!("GSPLAT_LOD_PX {v:?} is not a positive number"))?,
        },
        budget: match budget {
            None => default.budget,
            Some(v) => Some(
                v.parse::<u64>()
                    .map_err(|e| format!("GSPLAT_LOD_BUDGET {v:?} is not a row count ({e})"))?,
            ),
        },
        min_level: match min_level {
            None => default.min_level,
            Some(v) => v
                .parse::<u32>()
                .map_err(|e| format!("GSPLAT_LOD_MIN_LEVEL {v:?} is not a level ({e})"))?,
        },
    })
}

/// lod-meta.json facts by payload generation, kept while drawn; nothing is decoded here.
#[derive(Default)]
pub(crate) struct LodInfoCache(crate::generation_cache::GenerationCache<Arc<Info>>);

impl re_viewer_context::Cache for LodInfoCache {
    fn name(&self) -> &'static str {
        "SogLodSplats3D lod-meta"
    }

    fn begin_frame(&mut self) {
        self.0.begin_frame();
    }

    fn purge_memory(&mut self) {
        *self = Self::default();
    }
}

impl re_byte_size::MemUsageTreeCapture for LodInfoCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        // The parsed tree: a few dozen bytes per leaf.
        re_byte_size::MemUsageNode::new().into_tree()
    }
}

impl LodInfoCache {
    /// lod-meta.json + each file's meta.json, parsed once per `generation`.
    pub(crate) fn info(
        &mut self,
        generation: u64,
        lod_meta: &str,
        metas: &[&str],
    ) -> Result<Arc<Info>, String> {
        self.0
            .get_or_parse(generation, || info(lod_meta, metas).map(Arc::new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gsplat_core::lod::LodMode;

    #[test]
    fn parameters_parse_with_defaults_and_reject_bad_values() {
        let p = parse_params(None, None, None, None).unwrap();
        assert_eq!(p, LodParams::default());
        let p = parse_params(Some("cull"), Some("4.5"), Some("1000"), Some("1")).unwrap();
        assert_eq!(
            (p.mode, p.tau_px, p.budget, p.min_level),
            (LodMode::Cull, 4.5, Some(1000), 1)
        );
        assert!(parse_params(Some("fast"), None, None, None).is_err());
        assert!(parse_params(None, Some("-1"), None, None).is_err());
        assert!(parse_params(None, None, Some("x"), None).is_err());
    }

    #[test]
    fn planes_split_per_file_by_sizes() {
        let data: Vec<u8> = (0..20).collect();
        let sizes = [2, 1, 1, 1, 1, 0, 0, 3, 1, 1, 1, 1, 2, 2];
        let planes = [
            Some(&data[0..5]),
            Some(&data[5..7]),
            Some(&data[7..9]),
            Some(&data[9..11]),
            Some(&data[11..13]),
            Some(&data[13..15]),
            Some(&data[15..17]),
        ];
        let files = split_planes(2, &sizes, planes).unwrap();
        assert_eq!(files[0].means_l, &[0, 1]);
        assert_eq!(files[1].means_l, &[2, 3, 4]);
        assert!(files[0].sh_centroids.is_none());
        assert_eq!(files[1].sh_labels, Some(&data[15..17]));
        assert!(split_planes(3, &sizes, planes).is_err());
        let short = [2, 1, 1, 1, 1, 0, 0, 9, 1, 1, 1, 1, 2, 2];
        assert!(split_planes(2, &short, planes).is_err());
    }
}
