//! Streamed SOG sets on disk (`lod-meta.json` + one unbundled SOG directory per file) and the `--lod` flags.
use anyhow::{Context as _, Result};
use gsplat_core::lod::{
    LoadOptions, LodFile, LodIndex, LodMeta, LodMode, LodParams, LodSet, SelectionStats,
};
use gsplat_core::sog::SogWebp;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// `--lod` flags shared by `render` and `speed`.
#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct LodArgs {
    /// A Streamed SOG set: the directory of `lod-meta.json` (or the file). Replaces `--ply`.
    #[arg(long)]
    pub lod: Option<PathBuf>,
    /// `off` (all rows of the finest level, no cull), `cull` (finest level, leaves outside the frustum dropped) or
    /// `lod` (frustum cull + per-leaf level by the pixel rule).
    #[arg(long, default_value = "lod", value_parser = str::parse::<LodMode>)]
    pub lod_mode: LodMode,
    /// Pixel rule: per leaf the coarsest level whose feature size (`--lod-percentile` of its splats' 1-sigma major
    /// axes) projects to at most this many pixels.
    #[arg(long, default_value_t = LodParams::default().tau_px)]
    pub lod_px: f32,
    /// Optional splat budget (rows per view).
    #[arg(long)]
    pub lod_budget: Option<u64>,
    /// Drop levels finer than this at load.
    #[arg(long, default_value_t = 0)]
    pub lod_min_level: u32,
    /// Clamp opacity to 1 (single-SOG behaviour) instead of keeping merged v3 opacity up to 1.5625.
    #[arg(long)]
    pub lod_clamp_opacity: bool,
    /// Percentile (0..1) of a leaf's splat sizes that rates the leaf at a level.
    #[arg(long, default_value_t = LoadOptions::default().feature_percentile)]
    pub lod_percentile: f32,
}

impl LodArgs {
    pub fn params(&self) -> LodParams {
        LodParams {
            mode: self.lod_mode,
            tau_px: self.lod_px,
            budget: self.lod_budget,
            min_level: self.lod_min_level,
        }
    }

    pub fn load_options(&self) -> LoadOptions {
        LoadOptions {
            min_level: self.lod_min_level,
            clamp_opacity: self.lod_clamp_opacity,
            feature_percentile: self.lod_percentile,
        }
    }
}

/// The bytes of one SOG file.
pub struct FileBytes {
    meta_json: String,
    /// `means_l`, `means_u`, `quats`, `scales`, `sh0`, then optionally `shN_centroids`, `shN_labels`.
    planes: Vec<Vec<u8>>,
}

/// A set read from disk, with the selection parameters to render it with.
pub struct LodSource {
    pub meta: LodMeta,
    pub files: Vec<Option<FileBytes>>,
    pub load: LoadOptions,
    pub params: LodParams,
    pub dir: PathBuf,
    /// Seconds to read the files.
    pub read_seconds: f64,
}

fn plane_files(meta: &serde_json::Value) -> Result<Vec<String>> {
    let files = |key: &str| -> Option<Vec<String>> {
        meta.get(key)?.get("files")?.as_array().map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
    };
    let mut out = Vec::new();
    for key in ["means", "quats", "scales", "sh0"] {
        out.extend(files(key).with_context(|| format!("SOG meta.json has no {key}.files"))?);
    }
    // means files are [means_l, means_u]; the order of SogWebp is means_l, means_u, quats, scales, sh0.
    if let Some(sh) = files("shN") {
        out.extend(sh);
    }
    Ok(out)
}

impl LodSource {
    /// Read `lod-meta.json` and every file of a level >= `args.lod_min_level`.
    pub fn read(path: &Path, args: &LodArgs) -> Result<Self> {
        let start = std::time::Instant::now();
        let (dir, index) = if path.is_dir() {
            (path.to_path_buf(), path.join("lod-meta.json"))
        } else {
            (
                path.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf(),
                path.to_path_buf(),
            )
        };
        let meta = LodMeta::parse(
            &std::fs::read_to_string(&index).with_context(|| format!("{}", index.display()))?,
        )?;
        let load = args.load_options();
        let mut files: Vec<Option<FileBytes>> = std::iter::repeat_with(|| None)
            .take(meta.filenames.len())
            .collect();
        for (file, _) in meta.loaded_files(load.min_level)? {
            let meta_path = dir.join(&meta.filenames[file]);
            let file_dir = meta_path.parent().unwrap_or(&dir);
            let meta_json = std::fs::read_to_string(&meta_path)
                .with_context(|| format!("{}", meta_path.display()))?;
            let value: serde_json::Value = serde_json::from_str(&meta_json)?;
            let planes = plane_files(&value)?
                .iter()
                .map(|f| {
                    std::fs::read(file_dir.join(f))
                        .with_context(|| format!("{}", file_dir.join(f).display()))
                })
                .collect::<Result<Vec<_>>>()?;
            anyhow::ensure!(
                planes.len() == 5 || planes.len() == 7,
                "{}: expected 5 or 7 planes",
                meta_path.display()
            );
            files[file] = Some(FileBytes { meta_json, planes });
        }
        Ok(Self {
            meta,
            files,
            load,
            params: args.params(),
            dir,
            read_seconds: start.elapsed().as_secs_f64(),
        })
    }

    /// The set as the core reads it.
    pub fn set(&self) -> LodSet<'_> {
        LodSet {
            meta: &self.meta,
            files: self
                .files
                .iter()
                .map(|f| {
                    f.as_ref().map(|f| LodFile {
                        meta_json: &f.meta_json,
                        webp: SogWebp {
                            means_l: &f.planes[0],
                            means_u: &f.planes[1],
                            quats: &f.planes[2],
                            scales: &f.planes[3],
                            sh0: &f.planes[4],
                            sh_centroids: f.planes.get(5).map(Vec::as_slice),
                            sh_labels: f.planes.get(6).map(Vec::as_slice),
                        },
                    })
                })
                .collect(),
        }
    }

    /// Centre and half diagonal of the leaf boxes.
    pub fn bounds(&self) -> (glam::Vec3, f32) {
        let (mut min, mut max) = (glam::Vec3::INFINITY, glam::Vec3::NEG_INFINITY);
        for leaf in &self.meta.leaves {
            min = min.min(leaf.min.into());
            max = max.max(leaf.max.into());
        }
        ((min + max) * 0.5, (max - min).length() * 0.5)
    }
}

/// Per camera: what the selection drew and what the forward pass then counted.
#[derive(Clone, Debug, Serialize)]
pub struct CameraSelection {
    pub camera: usize,
    #[serde(flatten)]
    pub stats: SelectionStats,
    pub select_us: f64,
    pub visible: u32,
    pub intersections: u32,
}

impl CameraSelection {
    /// The last selection of `renderer` (a Streamed SOG renderer) with the counts of the frame it drew.
    pub fn new(
        camera: usize,
        renderer: &crate::StandaloneRenderer,
        visible: u32,
        intersections: u32,
    ) -> Result<Self> {
        let (stats, select_us) = renderer.lod_stats().context("no level selection")?;
        Ok(Self {
            camera,
            stats: stats.clone(),
            select_us,
            visible,
            intersections,
        })
    }
}

/// What a report says about the loaded set.
#[derive(Clone, Debug, Serialize)]
pub struct SetReport {
    pub dir: PathBuf,
    pub args: LodArgs,
    pub levels: u32,
    pub leaves: usize,
    pub scene_rows: u32,
    pub level_rows: Vec<u64>,
    pub feature_size: Vec<f32>,
    pub read_seconds: f64,
    pub upload_seconds: f64,
    pub scene_gpu_bytes: u64,
    pub view_gpu_bytes: u64,
}

impl SetReport {
    pub fn new(
        source: &LodSource,
        args: &LodArgs,
        renderer: &crate::StandaloneRenderer,
    ) -> Result<Self> {
        let index: &LodIndex = renderer
            .lod_index()
            .context("not a Streamed SOG renderer")?;
        let levels = index.levels();
        let (scene_gpu_bytes, view_gpu_bytes) = renderer.gpu_bytes();
        Ok(Self {
            dir: source.dir.clone(),
            args: args.clone(),
            levels,
            leaves: source.meta.leaves.len(),
            scene_rows: index.rows(),
            level_rows: (0..levels).map(|l| index.level_rows(l)).collect(),
            feature_size: (0..levels).map(|l| index.feature_size(l)).collect(),
            read_seconds: source.read_seconds,
            upload_seconds: renderer.lod_upload_seconds().unwrap_or(0.0),
            scene_gpu_bytes,
            view_gpu_bytes,
        })
    }
}
