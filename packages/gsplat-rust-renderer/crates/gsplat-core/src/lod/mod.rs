//! Level of detail for `PlayCanvas` Streamed SOG sets (`lod-meta.json` + one SOG per file).
//!
//! Spec: <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/> (version 1).
//! The tree holds boxes; each leaf holds, per level, one contiguous run of rows in one file. Level 0 is the finest.
//! The legacy per-leaf `errors` (`lodErrors`) are ignored: the pixel rule uses a feature size per level that the
//! loader measures from the data ([`LodIndex::new`]).
//!
//! All levels of all loaded files live in ONE [`crate::Scene`]; files are uploaded in (level, file) order, so the
//! finest loaded level is one run at row 0. [`select`] is a pure function of (index, camera, instance transform,
//! parameters) and returns global row runs; [`crate::ViewState::set_runs`] hands them to the forward projection.
mod meta;
mod select;
mod upload;

pub use meta::{Error, FileRun, LeafMeta, LodMeta};
pub use select::{LodMode, LodParams, Selection, SelectionStats, select};
pub(crate) use upload::upload;
pub use upload::{LoadOptions, LodFile, LodSet, fit_min_level};

use glam::Vec3;

/// Runs one view can take: a 64 KiB uniform of `vec4<u32>`, one header + two (first, prefix) pairs per entry.
pub const MAX_RUNS: usize = 2 * (RUN_UNIFORM_BYTES / 16 - 1);

/// Bytes of the run-list uniform (`project.wgsl` `RunList`); the WebGPU default `maxUniformBufferBindingSize`.
pub(crate) const RUN_UNIFORM_BYTES: usize = 65_536;

/// A contiguous range of global scene rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub first: u32,
    pub count: u32,
}

impl Run {
    pub fn end(&self) -> u32 {
        self.first + self.count
    }
}

/// A leaf of the index: box (object space) and, per level, its global rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Leaf {
    pub min: Vec3,
    pub max: Vec3,
    pub runs: Vec<Option<Run>>,
    /// Per level: the leaf's own feature size (a high percentile of its rows' 1-sigma major axis, object units);
    /// empty or NaN = use the level's [`LodIndex::feature_size`].
    pub feature: Vec<f32>,
    /// How far splats reach out of the box (object units): the largest 3-sigma major axis over the leaf's rows at
    /// every loaded level. The frustum cull tests the box grown by it, so a splat centred outside the frustum that
    /// reaches into the frame keeps its leaf.
    pub reach: f32,
}

/// Host-side index of an uploaded set.
#[derive(Clone, Debug)]
pub struct LodIndex {
    leaves: Vec<Leaf>,
    /// Per level: median splat size (1 sigma of the largest axis, object units); NaN where not loaded.
    feature: Vec<f32>,
    level_rows: Vec<u64>,
    rows: u32,
    /// Global row ranges per level (first, end, level), sorted by first.
    ranges: Vec<(u32, u32, u32)>,
}

impl LodIndex {
    /// An index from leaves (global runs per level), the feature size per level, and the level of each row range
    /// `(first, end, level)` of the scene. Runs must lie inside the ranges of their level.
    pub fn new(
        leaves: Vec<Leaf>,
        feature: Vec<f32>,
        mut ranges: Vec<(u32, u32, u32)>,
    ) -> Result<Self, Error> {
        ranges.sort_unstable();
        let levels = feature.len();
        let mut level_rows = vec![0u64; levels];
        for &(first, end, level) in &ranges {
            if end < first || level as usize >= levels {
                return Err(Error::Invalid(format!(
                    "row range {first}..{end} level {level}"
                )));
            }
            level_rows[level as usize] += u64::from(end - first);
        }
        let rows = ranges.iter().map(|r| r.1).max().unwrap_or(0);
        for leaf in &leaves {
            if leaf.runs.len() != levels {
                return Err(Error::Invalid("leaf level count".into()));
            }
            for (level, run) in leaf.runs.iter().enumerate() {
                if let Some(run) = run {
                    let inside = ranges
                        .iter()
                        .any(|&(f, e, l)| l as usize == level && run.first >= f && run.end() <= e);
                    if !inside {
                        return Err(Error::Invalid(format!(
                            "leaf run {run:?} is outside the rows of level {level}"
                        )));
                    }
                }
            }
        }
        Ok(Self {
            leaves,
            feature,
            level_rows,
            rows,
            ranges,
        })
    }

    pub fn levels(&self) -> u32 {
        self.feature.len() as u32
    }

    pub fn leaves(&self) -> &[Leaf] {
        &self.leaves
    }

    /// Rows in the scene (all loaded levels).
    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn level_rows(&self, level: u32) -> u64 {
        self.level_rows.get(level as usize).copied().unwrap_or(0)
    }

    /// Median splat size of a level (object units); NaN when the level is not loaded.
    pub fn feature_size(&self, level: u32) -> f32 {
        self.feature
            .get(level as usize)
            .copied()
            .unwrap_or(f32::NAN)
    }

    /// The finest loaded level.
    pub fn min_level(&self) -> u32 {
        (0..self.levels())
            .find(|&l| self.level_rows(l) > 0)
            .unwrap_or(0)
    }

    pub fn level_of_row(&self, row: u32) -> Option<u32> {
        let i = self.ranges.partition_point(|r| r.0 <= row);
        let &(first, end, level) = self.ranges.get(i.checked_sub(1)?)?;
        (row >= first && row < end).then_some(level)
    }
}

/// Sort runs by first row and merge touching or overlapping ones.
pub fn merge(mut runs: Vec<Run>) -> Vec<Run> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<Run> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.end() => {
                let end = last.end().max(run.end());
                last.count = end - last.first;
            }
            _ => out.push(run),
        }
    }
    out
}

/// The run-list uniform words: header (count, total, active, 0), then per run (first, prefix) pairs, two per vec4.
/// `None` = inactive (all rows, the old path).
pub(crate) fn run_words(runs: Option<&[Run]>) -> Vec<u32> {
    let Some(runs) = runs else {
        return vec![0; 4];
    };
    let mut words = vec![0u32; 4 + 2 * runs.len().div_ceil(2) * 2];
    let mut total = 0u32;
    for (i, run) in runs.iter().enumerate() {
        words[4 + 2 * i] = run.first;
        words[4 + 2 * i + 1] = total;
        total += run.count;
    }
    words[0] = runs.len() as u32;
    words[1] = total;
    words[2] = 1;
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::pinhole_camera;

    #[test]
    fn adjacent_runs_merge() {
        let runs = merge(vec![
            Run {
                first: 10,
                count: 5,
            },
            Run {
                first: 0,
                count: 10,
            },
            Run {
                first: 20,
                count: 1,
            },
        ]);
        assert_eq!(
            runs,
            vec![
                Run {
                    first: 0,
                    count: 15
                },
                Run {
                    first: 20,
                    count: 1
                }
            ]
        );
    }

    #[test]
    fn run_words_hold_prefix_sums() {
        let w = run_words(Some(&[
            Run { first: 7, count: 3 },
            Run {
                first: 20,
                count: 2,
            },
            Run {
                first: 30,
                count: 1,
            },
        ]));
        assert_eq!(&w[..4], &[3, 6, 1, 0]);
        assert_eq!(run_words(Some(&[])), vec![0, 0, 1, 0]);
        assert_eq!(run_words(None), vec![0; 4]);
        assert_eq!(&w[4..10], &[7, 0, 20, 3, 30, 5]);
        assert_eq!(w.len() % 4, 0);
        assert_eq!(MAX_RUNS, 8190);
    }

    /// Splats in front of a 64 px pinhole camera; row i is deterministic.
    fn cloud(n: usize) -> crate::Splats {
        let mut state = 12345u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32
        };
        let mut splats = crate::Splats::default();
        for _ in 0..n {
            let z = 2.0 + 4.0 * next();
            splats.transforms.push([
                (next() - 0.5) * z,
                (next() - 0.5) * z,
                z,
                1.0,
                0.0,
                0.0,
                0.0,
                -3.0 - next(),
                -3.0 - next(),
                -3.0 - next(),
            ]);
            splats.raw_opacities.push(next() * 4.0 - 1.0);
            splats.sh_coefficients.push([next(), next(), next()]);
        }
        splats
    }

    fn rows(splats: &crate::Splats, runs: &[Run]) -> crate::Splats {
        let mut out = crate::Splats::default();
        for run in runs {
            for i in run.first as usize..run.end() as usize {
                out.transforms.push(splats.transforms[i]);
                out.raw_opacities.push(splats.raw_opacities[i]);
                out.sh_coefficients.push(splats.sh_coefficients[i]);
            }
        }
        out
    }

    fn image(
        renderer: &crate::Renderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &std::sync::Arc<crate::Scene>,
        runs: Option<&[Run]>,
    ) -> (Vec<[f32; 4]>, crate::FrameStats) {
        let mut view = renderer.create_view(scene, 1).unwrap();
        view.set_runs(runs).unwrap();
        let target = crate::test_utils::upload(device, &vec![[0.0f32; 4]; 64 * 64]);
        let stats = crate::test_utils::render_fitted(
            device,
            queue,
            renderer,
            &mut view,
            &pinhole_camera(64),
            &crate::RenderOptions::default(),
            &crate::Target::Float(target.clone()),
        );
        (
            crate::test_utils::read(device, queue, &target, 64 * 64),
            stats,
        )
    }

    #[test]
    #[ignore = "integration: GPU"]
    fn a_run_list_renders_the_same_image_as_a_scene_of_those_rows() {
        let (device, queue) = &crate::test_utils::gpu();
        let renderer = crate::Renderer::new(device).unwrap();
        let splats = cloud(3000);
        let scene = renderer.upload(&splats).unwrap();
        // Runs that cross workgroup boundaries, a single row, and the last row.
        let runs = [
            Run {
                first: 3,
                count: 400,
            },
            Run {
                first: 700,
                count: 1,
            },
            Run {
                first: 1000,
                count: 1200,
            },
            Run {
                first: 2999,
                count: 1,
            },
        ];
        let (selected, stats) = image(&renderer, device, queue, &scene, Some(&runs));
        let subset = renderer.upload(&rows(&splats, &runs)).unwrap();
        let (expected, expected_stats) = image(&renderer, device, queue, &subset, None);
        assert_eq!(stats.visible, expected_stats.visible);
        assert_eq!(stats.intersections, expected_stats.intersections);
        let worst = selected
            .iter()
            .zip(&expected)
            .flat_map(|(a, b)| (0..4).map(move |c| (a[c] - b[c]).abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "max channel difference {worst}");
        // No run list = every row; an empty list draws nothing.
        let (all, all_stats) = image(&renderer, device, queue, &scene, None);
        let (whole, _) = image(
            &renderer,
            device,
            queue,
            &scene,
            Some(&[Run {
                first: 0,
                count: 3000,
            }]),
        );
        assert_eq!(all, whole);
        assert!(all_stats.visible > stats.visible);
        let (_, empty) = image(&renderer, device, queue, &scene, Some(&[]));
        assert_eq!(empty.visible, 0);
    }

    #[test]
    #[ignore = "integration: GPU"]
    fn set_runs_rejects_unsorted_or_outside_runs() {
        let (device, _) = &crate::test_utils::gpu();
        let renderer = crate::Renderer::new(device).unwrap();
        let scene = renderer.upload(&cloud(10)).unwrap();
        let mut view = renderer.create_view(&scene, 1).unwrap();
        assert!(view.set_runs(Some(&[Run { first: 5, count: 6 }])).is_err());
        assert!(
            view.set_runs(Some(&[
                Run { first: 5, count: 1 },
                Run { first: 0, count: 1 }
            ]))
            .is_err()
        );
        let many = vec![Run { first: 0, count: 0 }; MAX_RUNS + 1];
        assert!(view.set_runs(Some(&many)).is_err());
        view.set_runs(Some(&[
            Run { first: 0, count: 2 },
            Run { first: 9, count: 1 },
        ]))
        .unwrap();
        assert_eq!(view.selected_rows(), 3);
    }
}
