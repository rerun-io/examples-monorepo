//! The pure per-view selection: frustum cull, then per leaf the coarsest level by the pixel rule, merged into runs.
use super::{Leaf, LodIndex, MAX_RUNS, Run, merge};
use glam::{Affine3A, Vec3};
use serde::Serialize;

/// What [`select`] does. Text form (`FromStr`, `Display`, serde): `off`, `cull`, `lod`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LodMode {
    /// All rows of the finest allowed level, no cull (the "before").
    #[default]
    Off,
    /// Finest allowed level, leaves outside the frustum dropped.
    Cull,
    /// Frustum cull, then per leaf the coarsest level whose feature size projects to at most `tau_px`.
    Lod,
}

impl LodMode {
    pub const NAMES: [(&str, Self); 3] =
        [("off", Self::Off), ("cull", Self::Cull), ("lod", Self::Lod)];
}

impl std::str::FromStr for LodMode {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::NAMES
            .iter()
            .find(|(name, _)| *name == text)
            .map(|&(_, mode)| mode)
            .ok_or_else(|| format!("LOD mode {text:?} is not off|cull|lod"))
    }
}

impl std::fmt::Display for LodMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = Self::NAMES
            .iter()
            .find(|(_, m)| m == self)
            .map_or("", |(n, _)| n);
        f.write_str(name)
    }
}

/// Per-view selection parameters, shared by the CLI and the viewer ([`LodParams::default`] is the one default).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LodParams {
    pub mode: LodMode,
    /// Pixel rule threshold: the leaf's feature size (a percentile of its 1-sigma major axes, see
    /// [`LoadOptions::feature_percentile`]) projected to pixels.
    pub tau_px: f32,
    /// Optional splat budget (rows): `tau` grows until the selection fits (or every leaf is at its coarsest).
    pub budget: Option<u64>,
    /// Never pick a level finer than this.
    pub min_level: u32,
}

impl Default for LodParams {
    /// tau 1 px with the leaf p90 rule: the measured "no visible loss" setting (Casa Loma whole view 40.8 dB with
    /// 0.81 M of 5.0 M rows; up-close views at run-to-run noise).
    fn default() -> Self {
        Self {
            mode: LodMode::Lod,
            tau_px: 1.0,
            budget: None,
            min_level: 0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SelectionStats {
    pub rows: u64,
    pub runs: u32,
    pub rows_per_level: Vec<u64>,
    pub leaves_culled: u32,
    pub leaves_drawn: u32,
    /// The pixel threshold after the budget (= `tau_px` without a budget).
    pub tau_used: f32,
    /// The rule's selection merged to more than [`MAX_RUNS`] runs; this is the bounded fallback (frustum cull at the
    /// finest allowed level, else that whole level).
    pub over_run_limit: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selection {
    /// Sorted by `first`, adjacent runs merged.
    pub runs: Vec<Run>,
    pub stats: SelectionStats,
}

/// Near plane of the pinhole projection (`project.wgsl`).
const NEAR: f32 = 0.01;

/// Whether the box lies fully outside the camera's pinhole frustum (one plane rejects all eight corners).
/// Lenses other than pinhole are never culled.
fn outside_frustum(
    camera: &crate::Camera,
    view_from_obj: &glam::Mat4,
    min: Vec3,
    max: Vec3,
) -> bool {
    if camera.model != crate::CameraModel::Pinhole {
        return false;
    }
    let f = camera.focal();
    let size = camera.size.as_vec2();
    let c = camera.center_uv * size;
    let corners: [Vec3; 8] = std::array::from_fn(|i| {
        let p = Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        );
        view_from_obj.transform_point3(p)
    });
    // Inside: z >= NEAR, 0 <= f x / z + c <= size  <=>  f x + c z >= 0, f x + (c - size) z <= 0.
    let planes: [&dyn Fn(Vec3) -> bool; 5] = [
        &|p| p.z < NEAR,
        &|p| f.x * p.x + c.x * p.z < 0.0,
        &|p| f.x * p.x + (c.x - size.x) * p.z > 0.0,
        &|p| f.y * p.y + c.y * p.z < 0.0,
        &|p| f.y * p.y + (c.y - size.y) * p.z > 0.0,
    ];
    planes.iter().any(|out| corners.iter().all(|&p| out(p)))
}

/// Distance from `point` to the nearest point of the box (0 inside).
fn box_distance(point: Vec3, min: Vec3, max: Vec3) -> f32 {
    point.clamp(min, max).distance(point)
}

/// How much more a local length can grow than a local distance under `m`: largest over smallest singular value
/// (infinite when singular). Exactly 1 for a rotation times a uniform scale (within float noise), so those keep
/// the plain local-space rule.
fn anisotropy(m: glam::Mat3A) -> f32 {
    // Eigenvalues of the symmetric M^T M (closed form for 3x3) are the squared singular values.
    let a = glam::Mat3::from(m.transpose() * m);
    let (a11, a22, a33) = (a.x_axis.x, a.y_axis.y, a.z_axis.z);
    let p1 = a.y_axis.x.powi(2) + a.z_axis.x.powi(2) + a.z_axis.y.powi(2);
    let (hi, lo) = if p1 == 0.0 {
        let d = Vec3::new(a11, a22, a33);
        (d.max_element(), d.min_element())
    } else {
        let q = (a11 + a22 + a33) / 3.0;
        let p2 = (a11 - q).powi(2) + (a22 - q).powi(2) + (a33 - q).powi(2) + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        let b = (a - glam::Mat3::from_diagonal(Vec3::splat(q))) * (1.0 / p);
        let phi = (b.determinant() / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
        let hi = q + 2.0 * p * phi.cos();
        let lo = q + 2.0 * p * (phi + 2.0 * std::f32::consts::FRAC_PI_3).cos();
        (hi, lo)
    };
    let ratio = (hi.max(0.0) / lo.max(0.0)).sqrt();
    if ratio.is_nan() {
        f32::INFINITY
    } else if ratio < 1.0 + 1e-4 {
        1.0
    } else {
        ratio
    }
}

/// Per leaf the coarsest allowed level whose feature size (the leaf's own, else the level's) projects to at most
/// `tau` pixels at `distance`. A leaf nearer than its own box size (or containing the camera) stays at the finest.
/// Sizes, extents and distances are local; `anisotropy` (>= 1) scales the local lengths so the rule holds in world
/// space under a non-uniform instance transform (a conservative bound: largest stretch of a length over the
/// smallest stretch of the distance).
fn pick_level(
    index: &LodIndex,
    leaf: &Leaf,
    finest: u32,
    focal: f32,
    distance: f32,
    tau: f32,
    anisotropy: f32,
) -> Option<u32> {
    let loaded = |l: u32| leaf.runs[l as usize].is_some();
    let finest_loaded = (finest..index.levels()).find(|&l| loaded(l))?;
    if distance < anisotropy * (leaf.max - leaf.min).max_element() {
        return Some(finest_loaded);
    }
    for level in (finest_loaded..index.levels()).rev() {
        let size = leaf
            .feature
            .get(level as usize)
            .copied()
            .filter(|f| f.is_finite())
            .unwrap_or_else(|| index.feature_size(level));
        if loaded(level) && anisotropy * size * focal <= tau * distance {
            return Some(level);
        }
    }
    Some(finest_loaded)
}

/// The rows a view draws, at most [`MAX_RUNS`] runs. Pure and deterministic.
pub fn select(
    index: &LodIndex,
    camera: &crate::Camera,
    world_from_local: &Affine3A,
    params: &LodParams,
) -> Selection {
    let selection = select_by_rule(index, camera, world_from_local, params);
    if selection.runs.len() <= MAX_RUNS {
        return selection;
    }
    // Too many runs for one view: the finest allowed level, frustum-culled, else all of it (never overlapping levels).
    let finest = params.min_level.max(index.min_level());
    let mut bounded = select_by_rule(
        index,
        camera,
        world_from_local,
        &LodParams {
            mode: LodMode::Cull,
            ..*params
        },
    );
    if bounded.runs.len() > MAX_RUNS {
        let runs = merge(
            index
                .ranges
                .iter()
                .filter(|r| r.2 == finest)
                .map(|r| Run {
                    first: r.0,
                    count: r.1 - r.0,
                })
                .collect(),
        );
        let mut rows_per_level = vec![0; index.levels() as usize];
        rows_per_level[finest as usize] = index.level_rows(finest);
        bounded = Selection {
            stats: SelectionStats {
                rows: runs.iter().map(|r| u64::from(r.count)).sum(),
                runs: runs.len() as u32,
                rows_per_level,
                leaves_drawn: index.leaves.len() as u32,
                ..Default::default()
            },
            runs,
        };
    }
    bounded.stats.over_run_limit = true;
    bounded.stats.tau_used = selection.stats.tau_used;
    bounded
}

fn select_by_rule(
    index: &LodIndex,
    camera: &crate::Camera,
    world_from_local: &Affine3A,
    params: &LodParams,
) -> Selection {
    let finest = params.min_level.max(index.min_level());
    let view_from_obj = camera.view_from_world() * glam::Mat4::from(*world_from_local);
    let eye = world_from_local.inverse().transform_point3(camera.position);
    let focal = camera.focal().max_element();
    let anisotropy = anisotropy(world_from_local.matrix3);
    // Per leaf: culled, or its distance.
    let distances: Vec<Option<f32>> = index
        .leaves
        .iter()
        .map(|leaf| {
            let culled = params.mode != LodMode::Off
                && outside_frustum(
                    camera,
                    &view_from_obj,
                    leaf.min - leaf.reach,
                    leaf.max + leaf.reach,
                );
            (!culled).then(|| box_distance(eye, leaf.min, leaf.max))
        })
        .collect();
    let choose = |tau: f32| -> Vec<Option<u32>> {
        index
            .leaves
            .iter()
            .zip(&distances)
            .map(|(leaf, distance)| {
                let d = (*distance)?;
                match params.mode {
                    LodMode::Off | LodMode::Cull => {
                        pick_level(index, leaf, finest, focal, d, 0.0, anisotropy)
                    }
                    LodMode::Lod => pick_level(index, leaf, finest, focal, d, tau, anisotropy),
                }
            })
            .collect()
    };
    let rows = |levels: &[Option<u32>]| -> u64 {
        index
            .leaves
            .iter()
            .zip(levels)
            .filter_map(|(leaf, l)| leaf.runs[(*l)? as usize])
            .map(|r| u64::from(r.count))
            .sum()
    };
    let mut tau = params.tau_px.max(0.0);
    let mut levels = choose(tau);
    if let (LodMode::Lod, Some(budget)) = (params.mode, params.budget)
        && rows(&levels) > budget
    {
        // Rows do not grow with tau: bisect on log tau for the smallest tau that fits.
        let (mut lo, mut hi) = (tau.max(1e-3), tau.max(1e-3) * 2f32.powi(40));
        // Above `hi` nothing coarsens further: take it as it is.
        if rows(&choose(hi)) <= budget {
            for _ in 0..40 {
                let mid = (lo * hi).sqrt();
                if rows(&choose(mid)) > budget {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        tau = hi;
        levels = choose(tau);
    }
    let mut stats = SelectionStats {
        rows_per_level: vec![0; index.levels() as usize],
        tau_used: tau,
        ..Default::default()
    };
    let mut runs = Vec::new();
    for ((leaf, level), distance) in index.leaves.iter().zip(&levels).zip(&distances) {
        if distance.is_none() {
            stats.leaves_culled += 1;
            continue;
        }
        let Some(level) = *level else { continue };
        let Some(run) = leaf.runs[level as usize] else {
            continue;
        };
        stats.leaves_drawn += 1;
        stats.rows_per_level[level as usize] += u64::from(run.count);
        if run.count > 0 {
            runs.push(run);
        }
    }
    let runs = merge(runs);
    stats.rows = runs.iter().map(|r| u64::from(r.count)).sum();
    stats.runs = runs.len() as u32;
    Selection { runs, stats }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::pinhole_camera;

    /// Three levels; level k feature size 0.01 * 2^k. One leaf per x position along +z.
    fn index(zs: &[f32]) -> LodIndex {
        let counts = [100u32, 50, 25];
        let mut leaves = Vec::new();
        let mut next = [0u32, 10_000, 20_000];
        for &z in zs {
            let runs = (0..3)
                .map(|l| {
                    let run = Run {
                        first: next[l],
                        count: counts[l],
                    };
                    next[l] += counts[l];
                    Some(run)
                })
                .collect();
            leaves.push(Leaf {
                min: Vec3::new(-0.05, -0.05, z - 0.05),
                max: Vec3::new(0.05, 0.05, z + 0.05),
                runs,
                feature: Vec::new(),
                reach: 0.0,
            });
        }
        let ranges = (0..3).map(|l| (l * 10_000, next[l as usize], l)).collect();
        LodIndex::new(leaves, vec![0.01, 0.02, 0.04], ranges).unwrap()
    }

    fn camera() -> crate::Camera {
        pinhole_camera(1000) // fov 1 rad, focal ~ 915 px
    }

    fn lod(tau: f32) -> LodParams {
        LodParams {
            mode: LodMode::Lod,
            tau_px: tau,
            budget: None,
            min_level: 0,
        }
    }

    #[test]
    fn a_leaf_behind_the_camera_is_culled() {
        let index = index(&[-5.0, 5.0]);
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0));
        assert_eq!(s.stats.leaves_culled, 1);
        assert_eq!(s.stats.leaves_drawn, 1);
        let cull = LodParams {
            mode: LodMode::Cull,
            ..lod(2.0)
        };
        assert_eq!(
            select(&index, &camera(), &Affine3A::IDENTITY, &cull)
                .stats
                .rows,
            100
        );
        let off = LodParams {
            mode: LodMode::Off,
            ..lod(2.0)
        };
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &off);
        assert_eq!(
            (s.stats.rows, s.stats.leaves_culled, s.runs.len()),
            (200, 0, 1)
        );
    }

    #[test]
    fn a_leaf_beside_the_frustum_is_culled() {
        let mut index = index(&[5.0]);
        index.leaves[0].min.x += 20.0;
        index.leaves[0].max.x += 20.0;
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0));
        assert_eq!(s.stats.leaves_culled, 1);
        assert!(s.runs.is_empty());
    }

    #[test]
    fn a_leaf_across_the_near_plane_is_drawn_at_level_0() {
        // z in -0.5..0.5: the camera is inside the box.
        let index = index(&[0.0]);
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(100.0));
        assert_eq!(s.stats.rows_per_level, vec![100, 0, 0]);
        // Box from z = -0.4 to 0.6 seen from z = -1, crossing nothing but close: still drawn.
        let mut cam = camera();
        cam.position.z = -0.6;
        let s = select(&index, &cam, &Affine3A::IDENTITY, &lod(2.0));
        assert_eq!(s.stats.leaves_drawn, 1);
    }

    #[test]
    fn a_leaf_just_outside_the_frustum_with_a_splat_reaching_in_is_kept() {
        // fov 1 rad: at z = 5 the frame edge is at x ~ 2.73. Box x in 3.0..3.1 lies just outside.
        let mut index = index(&[5.0]);
        index.leaves[0].min.x = 3.0;
        index.leaves[0].max.x = 3.1;
        assert_eq!(
            select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0))
                .stats
                .leaves_culled,
            1
        );
        // One splat of 3-sigma reach 0.5 crosses into the frame: kept; the level choice uses the box itself.
        index.leaves[0].reach = 0.5;
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0));
        assert_eq!((s.stats.leaves_culled, s.stats.leaves_drawn), (0, 1));
        // Far outside, the same reach does not save it.
        index.leaves[0].min.x = 20.0;
        index.leaves[0].max.x = 20.1;
        assert_eq!(
            select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0))
                .stats
                .leaves_culled,
            1
        );
    }

    #[test]
    fn a_box_across_the_near_plane_beside_the_camera_is_drawn() {
        let mut index = index(&[0.0]);
        index.leaves[0].min = Vec3::new(0.2, -0.1, -1.0);
        index.leaves[0].max = Vec3::new(0.4, 0.1, 1.0);
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0));
        assert_eq!((s.stats.leaves_drawn, s.stats.rows_per_level[0]), (1, 100));
    }

    #[test]
    fn a_leaf_nearer_than_its_own_size_stays_at_level_0() {
        let mut index = index(&[20.0]);
        assert_eq!(
            select(&index, &camera(), &Affine3A::IDENTITY, &lod(100.0))
                .stats
                .rows_per_level,
            vec![0, 0, 25]
        );
        index.leaves[0].min = Vec3::new(-15.0, -15.0, 5.0);
        index.leaves[0].max = Vec3::new(15.0, 15.0, 35.0);
        assert_eq!(
            select(&index, &camera(), &Affine3A::IDENTITY, &lod(100.0))
                .stats
                .rows_per_level,
            vec![100, 0, 0]
        );
    }

    #[test]
    fn a_leaf_with_large_merged_splats_keeps_a_finer_level() {
        // At z = 30 the level features (0.01, 0.02, 0.04) project to ~0.3, 0.6, 1.2 px: tau 2 takes level 2.
        let mut index = index(&[30.0, 30.0]);
        // Leaf 1: level 2 projects to ~6 px, level 1 to ~1.5 px: level 1.
        index.leaves[1].feature = vec![0.01, 0.05, 0.2];
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0));
        assert_eq!(s.stats.rows_per_level, vec![0, 50, 25]);
    }

    /// A fixed tree: leaves along +z and beside it, some with their own (larger) features.
    fn pinned_index() -> LodIndex {
        let zs: Vec<f32> = (0..60).map(|i| 0.4 + i as f32 * 0.7).collect();
        let mut index = index(&zs);
        for (i, leaf) in index.leaves.iter_mut().enumerate() {
            let x = ((i * 7) % 5) as f32 - 2.0;
            leaf.min.x += x;
            leaf.max.x += x;
            if i % 3 == 0 {
                leaf.feature = vec![0.01, 0.03 + 0.001 * i as f32, 0.09];
            }
        }
        index
    }

    /// Selection per (transform, tau): rows per level, run count, and a checksum of the runs.
    fn pinned_selections() -> Vec<(Vec<u64>, u32, u64)> {
        let index = pinned_index();
        let transforms = [
            Affine3A::IDENTITY,
            Affine3A::from_scale_rotation_translation(
                Vec3::splat(2.5),
                glam::Quat::from_rotation_y(0.3),
                Vec3::new(0.5, -0.2, 3.0),
            ),
        ];
        let mut out = Vec::new();
        for transform in &transforms {
            for tau in [0.5, 1.0, 2.0, 4.0] {
                let s = select(&index, &camera(), transform, &lod(tau));
                let checksum = s.runs.iter().fold(0u64, |h, r| {
                    h.wrapping_mul(1_000_003)
                        .wrapping_add(u64::from(r.first) * 31 + u64::from(r.count))
                });
                out.push((s.stats.rows_per_level, s.stats.runs, checksum));
            }
        }
        out
    }

    /// Uniform-scale selections are pinned (measured numbers depend on them): any change to the rule must keep these.
    #[test]
    fn uniform_scale_selections_are_pinned() {
        let got = pinned_selections();
        // Taken at b0397776b (before the anisotropy bound): identity, then scale 2.5 + rotation; tau 0.5, 1, 2, 4.
        let pinned = vec![
            (vec![5200, 300, 0], 7, 11_115_980_694_083_833_058),
            (vec![3500, 850, 150], 25, 1_475_075_056_871_666_781),
            (vec![2500, 500, 575], 34, 13_344_158_549_874_761_516),
            (vec![700, 450, 1050], 17, 17_309_468_142_221_572_591),
            (vec![5000, 350, 0], 10, 5_302_883_205_041_834_608),
            (vec![3300, 850, 175], 27, 16_799_247_746_214_786_776),
            (vec![2100, 550, 625], 34, 4_258_537_392_382_773_351),
            (vec![500, 450, 1075], 20, 12_203_953_448_940_837_765),
        ];
        assert_eq!(got, pinned);
    }

    #[test]
    fn a_non_uniform_instance_scale_rates_the_stretched_size() {
        // Identity: level 2 at tau 2 (see above). diag(100, 1, 1) stretches every splat 100x along x while the
        // distance along z stays: the bound keeps the finest level.
        let index = index(&[30.0]);
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0));
        assert_eq!(s.stats.rows_per_level, vec![0, 0, 25]);
        let stretched = Affine3A::from_scale(Vec3::new(100.0, 1.0, 1.0));
        let s = select(&index, &camera(), &stretched, &lod(2.0));
        assert_eq!(s.stats.rows_per_level, vec![100, 0, 0]);
        assert!((anisotropy(stretched.matrix3) - 100.0).abs() < 0.01);
        let similar = Affine3A::from_scale_rotation_translation(
            Vec3::splat(7.0),
            glam::Quat::from_euler(glam::EulerRot::XYZ, 0.3, -1.1, 2.0),
            Vec3::ONE,
        );
        assert_eq!(anisotropy(similar.matrix3), 1.0);
        assert_eq!(anisotropy(glam::Mat3A::ZERO), f32::INFINITY);
    }

    #[test]
    fn too_many_runs_fall_back_to_a_bounded_selection() {
        // Odd leaves keep level 0 (large level-1 feature), even leaves take level 1: 10,000 disjoint runs.
        let n = 10_000u32;
        let leaves = (0..n)
            .map(|i| Leaf {
                min: Vec3::new(-0.05, -0.05, 29.95),
                max: Vec3::new(0.05, 0.05, 30.05),
                runs: vec![
                    Some(Run {
                        first: 4 * i,
                        count: 4,
                    }),
                    Some(Run {
                        first: 4 * n + 2 * i,
                        count: 2,
                    }),
                ],
                feature: vec![0.01, if i % 2 == 1 { 1.0 } else { 0.001 }],
                reach: 0.0,
            })
            .collect();
        let index = LodIndex::new(
            leaves,
            vec![0.01, 0.02],
            vec![(0, 4 * n, 0), (4 * n, 6 * n, 1)],
        )
        .unwrap();
        assert!(
            select_by_rule(&index, &camera(), &Affine3A::IDENTITY, &lod(1.0))
                .runs
                .len()
                > MAX_RUNS
        );
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(1.0));
        assert!(s.stats.over_run_limit);
        assert_eq!(
            s.runs,
            vec![Run {
                first: 0,
                count: 4 * n
            }]
        );
        assert_eq!(s.stats.rows_per_level, vec![u64::from(4 * n), 0]);
    }

    #[test]
    fn tau_zero_gives_level_0_everywhere() {
        let index = index(&[2.0, 50.0, 1000.0]);
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(0.0));
        assert_eq!(s.stats.rows_per_level, vec![300, 0, 0]);
        assert_eq!(
            s.runs,
            vec![Run {
                first: 0,
                count: 300
            }]
        );
    }

    #[test]
    fn level_is_monotonic_in_distance() {
        let zs: Vec<f32> = (1..200).map(|i| i as f32 * 2.0).collect();
        let index = index(&zs);
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &lod(2.0));
        let mut last = 0;
        for (leaf, z) in index.leaves.iter().zip(&zs) {
            let sel = select(
                &LodIndex::new(
                    vec![leaf.clone()],
                    index.feature.clone(),
                    index.ranges.clone(),
                )
                .unwrap(),
                &camera(),
                &Affine3A::IDENTITY,
                &lod(2.0),
            );
            let level = sel
                .stats
                .rows_per_level
                .iter()
                .position(|&r| r > 0)
                .unwrap();
            assert!(level >= last, "level {level} < {last} at z {z}");
            last = level;
        }
        assert_eq!(last, 2);
        assert!(s.stats.rows_per_level.iter().all(|&r| r > 0));
    }

    #[test]
    fn budget_is_respected_and_selection_is_deterministic() {
        let zs: Vec<f32> = (2..101).map(|i| i as f32 * 0.5).collect();
        let index = index(&zs);
        let free = select(&index, &camera(), &Affine3A::IDENTITY, &lod(1.0));
        for budget in [5_000u64, 3_000, 2_475] {
            let params = LodParams {
                budget: Some(budget),
                ..lod(1.0)
            };
            let a = select(&index, &camera(), &Affine3A::IDENTITY, &params);
            assert!(a.stats.rows <= budget, "{} > {budget}", a.stats.rows);
            assert!(a.stats.rows <= free.stats.rows);
            assert!(a.stats.tau_used >= 1.0);
            assert_eq!(a, select(&index, &camera(), &Affine3A::IDENTITY, &params));
        }
        // Below the coarsest selection: every leaf at its coarsest level.
        let params = LodParams {
            budget: Some(10),
            ..lod(1.0)
        };
        assert_eq!(
            select(&index, &camera(), &Affine3A::IDENTITY, &params)
                .stats
                .rows_per_level,
            vec![0, 0, 25 * 99]
        );
    }

    #[test]
    fn min_level_caps_the_finest_level() {
        let index = index(&[2.0]);
        let params = LodParams {
            min_level: 1,
            ..lod(0.0)
        };
        let s = select(&index, &camera(), &Affine3A::IDENTITY, &params);
        assert_eq!(s.stats.rows_per_level, vec![0, 50, 0]);
    }

    #[test]
    fn level_of_row_finds_the_range() {
        let index = index(&[1.0, 2.0]);
        assert_eq!(index.level_of_row(0), Some(0));
        assert_eq!(index.level_of_row(10_049), Some(1));
        assert_eq!(index.level_of_row(5_000), None);
        assert_eq!(index.rows(), 20_050);
    }
}
