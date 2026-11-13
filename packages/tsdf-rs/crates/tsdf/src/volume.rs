use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::changes::{ChangeTracker, DirtyMeshes};
use crate::error::TsdfError;
use crate::extract::{self, VolumeView};
use crate::frame::{Camera, DepthFrame, DepthRange};
use crate::geometry::{
    add, add_scaled, camera_to_world, checked_floor_i64, invert_3x3, length, multiply_3x3,
    rotation_column, transform_point, validate_length, validate_non_negative, validate_positive,
};
use crate::mesh::Mesh;

/// Voxels per block edge, as a loop counter that converts losslessly to
/// `usize` and `i64`.
pub(crate) const BLOCK_RESOLUTION_U16: u16 = 16;
pub(crate) const BLOCK_RESOLUTION: i64 = BLOCK_RESOLUTION_U16 as i64;
pub(crate) const BLOCK_RESOLUTION_USIZE: usize = BLOCK_RESOLUTION_U16 as usize;
const BLOCK_VOXEL_COUNT: usize =
    BLOCK_RESOLUTION_USIZE * BLOCK_RESOLUTION_USIZE * BLOCK_RESOLUTION_USIZE;
/// Pixels per discovery task: enough to amortize a task's block set.
const DISCOVERY_CHUNK_PIXELS: usize = 2_048;
/// Lowest block index: one block above `i64::MIN / 16`, so the gradient one
/// voxel below a block's first voxel stays inside `i64`.
pub(crate) const MIN_BLOCK_INDEX: i64 = i64::MIN / BLOCK_RESOLUTION + 1;
const MAX_BLOCK_INDEX: i64 = (i64::MAX - BLOCK_RESOLUTION) / BLOCK_RESOLUTION;

/// Integer index of a 16-by-16-by-16 voxel block: block `k` holds global voxel
/// coordinates `16 * k` to `16 * k + 15` on each axis.
pub type BlockKey = [i64; 3];

/// The allocated blocks by key.
pub(crate) type Blocks = FxHashMap<BlockKey, VoxelBlock>;

#[derive(Clone, Copy)]
struct PixelRay {
    x_over_z: f32,
    y_over_z: f32,
    distance_multiplier: f32,
}

/// Camera rays per pixel for one image size and intrinsic matrix.
struct RayCache {
    image_size: [usize; 2],
    intrinsics: [f32; 9],
    rays: Vec<PixelRay>,
}

impl RayCache {
    fn new(image_size: [usize; 2], intrinsics: [f32; 9]) -> Result<Self, TsdfError> {
        let inverse =
            invert_3x3(intrinsics).ok_or(TsdfError::SingularMatrix { name: "intrinsics" })?;
        let [height, width] = image_size;
        let rays = (0..height * width)
            .into_par_iter()
            .map(|index| {
                let pixel = [(index % width) as f32, (index / width) as f32, 1.0];
                let ray = multiply_3x3(inverse, pixel);
                if ray[2] == 0.0 || !ray.iter().all(|value| value.is_finite()) {
                    return PixelRay {
                        x_over_z: f32::NAN,
                        y_over_z: f32::NAN,
                        distance_multiplier: f32::NAN,
                    };
                }
                PixelRay {
                    x_over_z: ray[0] / ray[2],
                    y_over_z: ray[1] / ray[2],
                    distance_multiplier: length(ray) / ray[2].abs(),
                }
            })
            .collect();
        Ok(Self {
            image_size,
            intrinsics,
            rays,
        })
    }
}

/// The rays of `cache`, rebuilt first if the image size or intrinsics changed.
fn cached_rays(
    cache: &mut Option<RayCache>,
    image_size: [usize; 2],
    intrinsics: [f32; 9],
) -> Result<&[PixelRay], TsdfError> {
    let cache = match cache.take() {
        Some(current) if current.image_size == image_size && current.intrinsics == intrinsics => {
            cache.insert(current)
        }
        _ => cache.insert(RayCache::new(image_size, intrinsics)?),
    };
    Ok(&cache.rays)
}

/// Running color means of a block's voxels.
pub(crate) struct ColorVoxels {
    pub(crate) rgb: Box<[[f32; 3]]>,
    /// Color observations per voxel; frames without color add none.
    weight: Box<[f32]>,
}

pub(crate) struct VoxelBlock {
    pub(crate) tsdf: Box<[f32]>,
    pub(crate) weights: Box<[f32]>,
    pub(crate) color: Option<ColorVoxels>,
    pub(crate) observed_min: f32,
    pub(crate) observed_max: f32,
}

impl VoxelBlock {
    pub(crate) fn new(fuse_color: bool) -> Self {
        Self {
            tsdf: vec![1.0; BLOCK_VOXEL_COUNT].into_boxed_slice(),
            weights: vec![0.0; BLOCK_VOXEL_COUNT].into_boxed_slice(),
            color: fuse_color.then(|| ColorVoxels {
                rgb: vec![[0.0; 3]; BLOCK_VOXEL_COUNT].into_boxed_slice(),
                weight: vec![0.0; BLOCK_VOXEL_COUNT].into_boxed_slice(),
            }),
            observed_min: f32::INFINITY,
            observed_max: f32::NEG_INFINITY,
        }
    }
}

/// Index of block-local voxel `[x, y, z]` (each in `0..16`) in a block's
/// arrays: x fastest. The one definition of the voxel layout.
pub(crate) fn voxel_index([x, y, z]: [usize; 3]) -> usize {
    (z * BLOCK_RESOLUTION_USIZE + y) * BLOCK_RESOLUTION_USIZE + x
}

/// [`voxel_index`] of a block-local coordinate held as `i64` (each in `0..16`).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "block-local coordinates are in 0..16"
)]
pub(crate) fn local_index(local: [i64; 3]) -> usize {
    voxel_index(local.map(|coordinate| coordinate as usize))
}

/// `key + offset`, when no coordinate overflows.
pub(crate) fn checked_offset(key: [i64; 3], offset: [i64; 3]) -> Option<[i64; 3]> {
    Some([
        key[0].checked_add(offset[0])?,
        key[1].checked_add(offset[1])?,
        key[2].checked_add(offset[2])?,
    ])
}

/// Global voxel coordinate of the first voxel of block `key`, when every
/// voxel of the block has an `i64` coordinate.
pub(crate) fn checked_block_origin(key: BlockKey) -> Option<[i64; 3]> {
    let origin = [
        key[0].checked_mul(BLOCK_RESOLUTION)?,
        key[1].checked_mul(BLOCK_RESOLUTION)?,
        key[2].checked_mul(BLOCK_RESOLUTION)?,
    ];
    checked_offset(origin, [BLOCK_RESOLUTION; 3])?;
    Some(origin)
}

/// World position of the center of global voxel `coordinate`.
pub(crate) fn voxel_center(coordinate: [i64; 3], voxel_size: f32) -> [f32; 3] {
    coordinate.map(|value| (value as f32 + 0.5) * voxel_size)
}

/// The running mean of `weight` observations with `mean`, after one more:
/// `observation`.
fn running_mean(mean: f32, weight: f32, observation: f32) -> f32 {
    (mean * weight + observation) / (weight + 1.0)
}

impl std::fmt::Debug for TsdfVolume {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TsdfVolume")
            .field("voxel_size", &self.voxel_size)
            .field("sdf_trunc", &self.sdf_trunc)
            .field("fuse_color", &self.fuse_color)
            .field("block_count", &self.blocks.len())
            .field("max_blocks", &self.max_blocks)
            .field("budget_reached", &self.budget_reached)
            .field("min_weight", &self.min_weight)
            .finish_non_exhaustive()
    }
}

/// Sparse, unbounded truncated signed-distance volume.
///
/// The volume allocates 16-by-16-by-16 voxel blocks around observed depth
/// surfaces. Voxels are sampled at their centers: global voxel `(x, y, z)`
/// sits at `((x, y, z) + 0.5) * voxel_size`. Depths and camera translations
/// use the same world units as `voxel_size` and `sdf_trunc`.
///
/// # Example
///
/// ```
/// use tsdf::{Camera, DepthFrame, TsdfVolume};
///
/// let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
/// let depth = vec![1.0_f32; 64 * 64];
/// let camera = Camera::pinhole([60.0, 60.0, 31.5, 31.5], Camera::IDENTITY);
/// volume.integrate(&DepthFrame::new(&depth, [64, 64]), &camera)?;
/// let mesh = volume.extract_mesh()?;
/// assert!(!mesh.triangles.is_empty());
/// # Ok::<(), tsdf::TsdfError>(())
/// ```
pub struct TsdfVolume {
    voxel_size: f32,
    sdf_trunc: f32,
    fuse_color: bool,
    blocks: Blocks,
    changes: ChangeTracker,
    ray_cache: Option<RayCache>,
    max_blocks: Option<usize>,
    budget_reached: bool,
    min_weight: f32,
}

impl TsdfVolume {
    /// Create an empty volume.
    ///
    /// `voxel_size` is the voxel edge and `sdf_trunc` the signed-distance
    /// truncation, both in world units. Observation weights accumulate as
    /// `f32`. `fuse_color` allocates and averages per-voxel RGB.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] if either distance is not finite and
    /// positive.
    pub fn new(voxel_size: f32, sdf_trunc: f32, fuse_color: bool) -> Result<Self, TsdfError> {
        validate_positive("voxel_size", voxel_size)?;
        validate_positive("sdf_trunc", sdf_trunc)?;
        Ok(Self {
            voxel_size,
            sdf_trunc,
            fuse_color,
            blocks: Blocks::default(),
            changes: ChangeTracker::default(),
            ray_cache: None,
            max_blocks: None,
            budget_reached: false,
            min_weight: 0.0,
        })
    }

    /// Cap the number of allocated blocks (`None`, the default: unbounded).
    ///
    /// Once the cap is reached, existing blocks keep fusing and new blocks
    /// are dropped (in sorted key order) until [`TsdfVolume::evict_farthest`]
    /// frees room. A memory bound for long sessions on small devices. Only
    /// new blocks that a frame actually updates count: the cap applies after
    /// integration, so a frame briefly holds its dropped blocks.
    pub fn set_max_blocks(&mut self, max_blocks: Option<usize>) {
        self.max_blocks = max_blocks;
    }

    /// Whether an integration skipped a new block because of the cap set
    /// with [`TsdfVolume::set_max_blocks`], since the last eviction.
    pub fn budget_reached(&self) -> bool {
        self.budget_reached
    }

    /// Number of allocated blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Free the blocks farthest from `center` (world units, block centers)
    /// until at most `keep` remain, and clear [`TsdfVolume::budget_reached`].
    ///
    /// An evicted block is forgotten: it is not reported by
    /// [`TsdfVolume::dirty_block_meshes`] (consumers clear its last mesh
    /// themselves), and a later frame that reaches it starts it afresh and
    /// delivers its mesh again. Kept neighbors of an evicted block are
    /// re-meshed: their cells and normals read its voxels, so their next
    /// meshes leave those out.
    /// Returns the evicted keys, farthest first.
    pub fn evict_farthest(&mut self, center: [f32; 3], keep: usize) -> Vec<BlockKey> {
        if self.blocks.len() <= keep {
            self.budget_reached = false;
            return Vec::new();
        }
        let block_length = self.voxel_size * BLOCK_RESOLUTION as f32;
        let mut by_distance: Vec<(f32, BlockKey)> = self
            .blocks
            .keys()
            .map(|key| {
                let distance = (0..3)
                    .map(|axis| ((key[axis] as f32 + 0.5) * block_length - center[axis]).powi(2))
                    .sum();
                (distance, *key)
            })
            .collect();
        // Farthest first; ties by key, so the result is deterministic.
        by_distance.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let evicted: Vec<BlockKey> = by_distance
            .into_iter()
            .take(self.blocks.len() - keep)
            .map(|(_, key)| key)
            .collect();
        // Freed keys are dropped from the dirty set below; the kept neighbors stay dirty.
        self.changes.mark_touched(&evicted);
        for key in &evicted {
            self.blocks.remove(key);
            self.changes.forget(key);
        }
        self.budget_reached = false;
        evicted
    }

    /// Skip mesh cells with a corner voxel fused fewer than `min_weight`
    /// times (0, the default: any observed voxel), so meshes hide
    /// single-view slivers. Applies to [`TsdfVolume::extract_mesh`] and
    /// [`TsdfVolume::dirty_block_meshes`] alike; a new value stales every
    /// block's mesh.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] if `min_weight` is not finite and
    /// non-negative.
    pub fn set_min_weight(&mut self, min_weight: f32) -> Result<(), TsdfError> {
        validate_non_negative("min_weight", min_weight)?;
        if min_weight != self.min_weight {
            self.min_weight = min_weight;
            self.changes.mark_all(self.blocks.keys());
        }
        Ok(())
    }

    /// Fuse one projective depth frame into the volume.
    ///
    /// Each voxel in front of the camera projects to its nearest pixel; a
    /// valid depth there updates the voxel's running TSDF mean, and the
    /// pixel's color its running color mean. Pixels outside
    /// [`DepthFrame::depth_range`], non-finite or non-positive are skipped.
    /// Reusing the image size and intrinsics reuses the per-pixel rays.
    ///
    /// # Errors
    ///
    /// [`TsdfError::ShapeMismatch`] when a slice length does not match the
    /// image size, [`TsdfError::DimensionOverflow`] when the image size
    /// overflows, [`TsdfError::InvalidParameter`] for an invalid depth range,
    /// and [`TsdfError::SingularMatrix`] when the intrinsics cannot be
    /// inverted.
    pub fn integrate(&mut self, frame: &DepthFrame<'_>, camera: &Camera) -> Result<(), TsdfError> {
        let [height, width] = frame.image_size;
        validate_length("depth", frame.depth.len(), &[height, width])?;
        if let Some(color) = frame.color {
            validate_length("color", color.len(), &[height, width, 3])?;
        }
        frame.depth_range.validate()?;
        let rays = cached_rays(&mut self.ray_cache, frame.image_size, camera.intrinsics)?;

        let context = FrameContext {
            depth: frame.depth,
            color: frame.color,
            depth_range: frame.depth_range,
            rays,
            width,
            height,
            intrinsics: camera.intrinsics,
            camera_from_world: camera.camera_from_world,
            voxel_size: self.voxel_size,
            sdf_trunc: self.sdf_trunc,
            min_weight: self.min_weight,
        };
        let allocated = self.blocks.len();
        let mut active: Vec<ActiveBlock> = discover_touched_blocks(&context)
            .into_iter()
            .map(|key| match self.blocks.remove(&key) {
                Some(block) => ActiveBlock {
                    key,
                    block,
                    is_new: false,
                    update: BlockUpdate::default(),
                },
                None => ActiveBlock {
                    key,
                    block: VoxelBlock::new(self.fuse_color),
                    is_new: true,
                    update: BlockUpdate::default(),
                },
            })
            .collect();
        active.par_iter_mut().for_each(|active| {
            active.update = integrate_block(active.key, &mut active.block, &context);
        });
        if let Some(limit) = self.max_blocks {
            self.budget_reached |=
                drop_new_blocks_over(limit.saturating_sub(allocated), &mut active);
        }
        // A new block that no voxel update reached holds nothing to keep.
        active.retain(|active| active.update.fused || !active.is_new);
        for active in &active {
            if let Some(changed) = &active.update.changed {
                self.changes.mark_changed(active.key, changed);
            }
        }
        self.blocks
            .extend(active.into_iter().map(|active| (active.key, active.block)));
        Ok(())
    }

    /// Extract the mesh of every observed zero crossing.
    ///
    /// Vertices lie on cell edges whose eight-corner cell is fully observed
    /// (see [`TsdfVolume::set_min_weight`]);
    /// each carries the color of its nearest voxel and the unit TSDF gradient
    /// as normal. Blocks are processed in sorted order and merged
    /// sequentially, so the bytes do not depend on the Rayon thread count.
    ///
    /// # Errors
    ///
    /// [`TsdfError::VertexIndexOverflow`] if the mesh exceeds 32-bit indices.
    pub fn extract_mesh(&self) -> Result<Mesh, TsdfError> {
        extract::full_mesh(&self.view())
    }

    /// Per-block meshes of the blocks whose surface changed since the last
    /// confirmed [`crate::Delivery`].
    ///
    /// Integration stales the meshes that its voxel changes can alter (the
    /// changed blocks' own, and their neighbors' across the faces the changes
    /// reach); a stale block whose new mesh is byte-identical to its last
    /// confirmed one (by a 64-bit hash of positions, triangles, colors and
    /// normals) is left out. Each mesh holds the triangles of cells anchored in its block, with
    /// vertices deduplicated within the block. The union of the latest mesh
    /// per key equals [`TsdfVolume::extract_mesh`] triangle for triangle, and
    /// a seam vertex has the same position, color and normal in every block
    /// that holds it. A block whose surface vanished since it last had
    /// triangles is reported once with an empty mesh so consumers can clear
    /// it; blocks that never had surface are omitted. Keys are sorted.
    ///
    /// The meshes stay pending until [`crate::Delivery::confirm`]; drop the
    /// delivery instead to get them again from the next call.
    pub fn dirty_block_meshes(&mut self) -> DirtyMeshes<'_> {
        let keys = self.changes.dirty_keys(|key| self.blocks.contains_key(key));
        let meshes = extract::block_meshes(&self.view(), &keys, &self.changes);
        self.changes.pending(meshes)
    }

    fn view(&self) -> VolumeView<'_> {
        VolumeView {
            blocks: &self.blocks,
            voxel_size: self.voxel_size,
            fuse_color: self.fuse_color,
            min_weight: self.min_weight,
        }
    }

    /// [`TsdfVolume::dirty_block_meshes`], confirmed at once.
    ///
    /// # Example
    ///
    /// ```
    /// use tsdf::{Camera, DepthFrame, TsdfVolume};
    ///
    /// let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    /// let depth = vec![1.0_f32; 64 * 64];
    /// let camera = Camera::pinhole([60.0, 60.0, 31.5, 31.5], Camera::IDENTITY);
    /// volume.integrate(&DepthFrame::new(&depth, [64, 64]), &camera)?;
    /// assert!(!volume.extract_dirty_block_meshes().is_empty());
    /// assert!(volume.extract_dirty_block_meshes().is_empty());
    /// # Ok::<(), tsdf::TsdfError>(())
    /// ```
    pub fn extract_dirty_block_meshes(&mut self) -> Vec<(BlockKey, Mesh)> {
        let DirtyMeshes { meshes, delivery } = self.dirty_block_meshes();
        delivery.confirm();
        meshes
    }
}

/// Everything one frame's integration reads, built once per frame.
struct FrameContext<'a> {
    depth: &'a [f32],
    color: Option<&'a [u8]>,
    depth_range: DepthRange,
    rays: &'a [PixelRay],
    width: usize,
    height: usize,
    intrinsics: [f32; 9],
    camera_from_world: [f32; 16],
    voxel_size: f32,
    sdf_trunc: f32,
    min_weight: f32,
}

/// Blocks within `sdf_trunc` of a back-projected valid depth sample.
fn discover_touched_blocks(context: &FrameContext<'_>) -> FxHashSet<BlockKey> {
    let blocks_per_unit = 1.0 / (f64::from(context.voxel_size) * BLOCK_RESOLUTION as f64);
    let sdf_trunc = f64::from(context.sdf_trunc);
    context
        .depth
        .par_chunks(DISCOVERY_CHUNK_PIXELS)
        .zip(context.rays.par_chunks(DISCOVERY_CHUNK_PIXELS))
        .map(|(depth_chunk, ray_chunk)| {
            let mut local = FxHashSet::default();
            // A chunk of neighboring pixels covers a few dozen blocks.
            local.reserve(32);
            let mut previous = None;
            for (&depth, ray) in depth_chunk.iter().zip(ray_chunk) {
                if !context.depth_range.contains(depth)
                    || !ray.x_over_z.is_finite()
                    || !ray.y_over_z.is_finite()
                {
                    continue;
                }
                let camera_point = [ray.x_over_z * depth, ray.y_over_z * depth, depth];
                let world_point = camera_to_world(context.camera_from_world, camera_point);
                if !world_point.iter().all(|coordinate| coordinate.is_finite()) {
                    continue;
                }
                let Some(bounds) = block_bounds(world_point, sdf_trunc, blocks_per_unit) else {
                    continue;
                };
                // Neighboring pixels nearly always cover the same blocks.
                if previous == Some(bounds) {
                    continue;
                }
                previous = Some(bounds);
                let (min, max) = bounds;
                for x in min[0]..=max[0] {
                    for y in min[1]..=max[1] {
                        for z in min[2]..=max[2] {
                            local.insert([x, y, z]);
                        }
                    }
                }
            }
            local
        })
        .reduce(FxHashSet::default, |mut left, mut right| {
            if left.len() < right.len() {
                std::mem::swap(&mut left, &mut right);
            }
            left.extend(right);
            left
        })
}

/// A block one frame's integration reaches.
struct ActiveBlock {
    key: BlockKey,
    block: VoxelBlock,
    /// Allocated by this integration.
    is_new: bool,
    update: BlockUpdate,
}

/// What one frame's integration did to a block.
#[derive(Default)]
struct BlockUpdate {
    /// Whether any voxel fused an observation.
    fused: bool,
    /// The voxels whose change can alter a mesh, if any.
    changed: Option<LocalBox>,
}

/// Inclusive bounds of block-local voxel coordinates, per axis.
pub(crate) struct LocalBox {
    pub(crate) min: [usize; 3],
    pub(crate) max: [usize; 3],
}

impl LocalBox {
    fn include(changed: &mut Option<Self>, local: [usize; 3]) {
        match changed {
            Some(bounds) => {
                bounds.min = std::array::from_fn(|axis| bounds.min[axis].min(local[axis]));
                bounds.max = std::array::from_fn(|axis| bounds.max[axis].max(local[axis]));
            }
            None => {
                *changed = Some(Self {
                    min: local,
                    max: local,
                });
            }
        }
    }
}

/// What fusing one observation did to a voxel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VoxelUpdate {
    /// No observation: outside the image or the depth range, or behind the
    /// truncation band.
    Skipped,
    /// Fused, leaving everything a mesh reads bit for bit as it was.
    Fused,
    /// Fused, and a mesh reading the voxel can change.
    Changed,
}

/// Clear the update, so they are dropped, of the updated new blocks beyond
/// the first `room` in sorted key order; whether any was.
fn drop_new_blocks_over(room: usize, active: &mut [ActiveBlock]) -> bool {
    let mut new: Vec<&mut ActiveBlock> = active
        .iter_mut()
        .filter(|active| active.is_new && active.update.fused)
        .collect();
    if new.len() <= room {
        return false;
    }
    new.sort_unstable_by_key(|active| active.key);
    for active in &mut new[room..] {
        active.update = BlockUpdate::default();
    }
    true
}

fn block_bounds(
    world_point: [f32; 3],
    sdf_trunc: f64,
    blocks_per_unit: f64,
) -> Option<(BlockKey, BlockKey)> {
    let block_index = |offset: f64| -> Option<BlockKey> {
        let mut key = [0; 3];
        for (index, &coordinate) in key.iter_mut().zip(&world_point) {
            *index = checked_floor_i64((f64::from(coordinate) + offset) * blocks_per_unit)
                .filter(|index| (MIN_BLOCK_INDEX..=MAX_BLOCK_INDEX).contains(index))?;
        }
        Some(key)
    };
    Some((block_index(-sdf_trunc)?, block_index(sdf_trunc)?))
}

/// Fuse the frame into one block.
fn integrate_block(
    key: BlockKey,
    block: &mut VoxelBlock,
    context: &FrameContext<'_>,
) -> BlockUpdate {
    let mut update = BlockUpdate::default();
    let Some(block_origin) = checked_block_origin(key) else {
        return update;
    };
    let voxel_size = context.voxel_size;
    let transform = context.camera_from_world;
    let first_camera = transform_point(transform, voxel_center(block_origin, voxel_size));
    // One voxel along each world axis, in camera coordinates.
    let [step_x, step_y, step_z] =
        [0, 1, 2].map(|axis| rotation_column(transform, axis).map(|value| value * voxel_size));

    for z in 0..BLOCK_RESOLUTION_USIZE {
        let camera_at_z = add_scaled(first_camera, step_z, z as f32);
        for y in 0..BLOCK_RESOLUTION_USIZE {
            let mut camera = add_scaled(camera_at_z, step_y, y as f32);
            for x in 0..BLOCK_RESOLUTION_USIZE {
                let index = voxel_index([x, y, z]);
                match integrate_voxel(index, camera, block, context) {
                    VoxelUpdate::Skipped => {}
                    VoxelUpdate::Fused => update.fused = true,
                    VoxelUpdate::Changed => {
                        update.fused = true;
                        LocalBox::include(&mut update.changed, [x, y, z]);
                    }
                }
                camera = add(camera, step_x);
            }
        }
    }
    update
}

/// Fuse the frame into voxel `index`, seen at camera point `camera`.
///
/// A mesh reads a voxel's TSDF value, whether its weight is zero or below
/// `min_weight`, and its color: the update is [`VoxelUpdate::Changed`] when
/// any of those can differ afterwards.
fn integrate_voxel(
    index: usize,
    camera: [f32; 3],
    block: &mut VoxelBlock,
    context: &FrameContext<'_>,
) -> VoxelUpdate {
    if camera[2] <= 0.0 || !camera[2].is_finite() {
        return VoxelUpdate::Skipped;
    }
    let pixel = multiply_3x3(context.intrinsics, camera);
    if pixel[2] == 0.0 || !pixel[2].is_finite() {
        return VoxelUpdate::Skipped;
    }
    let (Some(u), Some(v)) = (
        nearest_pixel(pixel[0] / pixel[2], context.width),
        nearest_pixel(pixel[1] / pixel[2], context.height),
    ) else {
        return VoxelUpdate::Skipped;
    };

    let pixel_index = v * context.width + u;
    let depth = context.depth[pixel_index];
    if !context.depth_range.contains(depth) {
        return VoxelUpdate::Skipped;
    }
    let distance_multiplier = context.rays[pixel_index].distance_multiplier;
    if !distance_multiplier.is_finite() {
        return VoxelUpdate::Skipped;
    }
    let signed_distance = (depth - camera[2]) * distance_multiplier;
    if signed_distance < -context.sdf_trunc {
        return VoxelUpdate::Skipped;
    }

    let observation = (signed_distance / context.sdf_trunc).min(1.0);
    let old_weight = block.weights[index];
    let updated_tsdf = running_mean(block.tsdf[index], old_weight, observation);
    let mut changed = updated_tsdf.to_bits() != block.tsdf[index].to_bits()
        || old_weight == 0.0
        || old_weight < context.min_weight;
    block.tsdf[index] = updated_tsdf;
    block.weights[index] = old_weight + 1.0;
    block.observed_min = block.observed_min.min(updated_tsdf);
    block.observed_max = block.observed_max.max(updated_tsdf);

    if let (Some(image), Some(color)) = (context.color, block.color.as_mut()) {
        let old_weight = color.weight[index];
        let old_rgb = color.rgb[index];
        let pixel = &image[pixel_index * 3..pixel_index * 3 + 3];
        color.rgb[index] = std::array::from_fn(|channel| {
            running_mean(old_rgb[channel], old_weight, f32::from(pixel[channel]))
        });
        color.weight[index] = old_weight + 1.0;
        changed |= color.rgb[index].map(f32::to_bits) != old_rgb.map(f32::to_bits);
    }
    if changed {
        VoxelUpdate::Changed
    } else {
        VoxelUpdate::Fused
    }
}

/// The pixel `x.round()` (half away from zero) selects on an axis of `size`
/// pixels, if inside, without the libm `round` call: in `f64`, `x + 0.5`
/// never rounds across an integer for an `f32` `x`, so truncating it rounds.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "positive here; the cast saturates and the bound rejects what it clamps"
)]
fn nearest_pixel(x: f32, size: usize) -> Option<usize> {
    let shifted = f64::from(x) + 0.5;
    // round() sends x <= -0.5 below zero; the cast saturates infinity, which
    // the bound then rejects.
    if shifted.is_nan() || shifted <= 0.0 {
        return None;
    }
    let index = shifted as usize;
    (index < size).then_some(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_first_observation_changes_the_voxel_even_when_its_value_does_not() {
        // A voxel 1 m in front of the surface observes the truncated value
        // 1.0, the value a new block starts with: only its weight changes.
        let intrinsics = Camera::pinhole([1_000.0, 1_000.0, 0.0, 0.0], Camera::IDENTITY).intrinsics;
        let rays = RayCache::new([1, 1], intrinsics).expect("invertible intrinsics");
        let context = FrameContext {
            depth: &[2.0],
            color: None,
            depth_range: DepthRange::default(),
            rays: &rays.rays,
            width: 1,
            height: 1,
            intrinsics,
            camera_from_world: Camera::IDENTITY,
            voxel_size: 0.04,
            sdf_trunc: 0.12,
            min_weight: 0.0,
        };
        let mut block = VoxelBlock::new(false);
        let fuse = |block: &mut VoxelBlock| integrate_voxel(0, [0.0, 0.0, 1.0], block, &context);
        assert!(fuse(&mut block) == VoxelUpdate::Changed);
        assert_eq!(block.tsdf[0].to_bits(), 1.0_f32.to_bits());
        assert!(fuse(&mut block) == VoxelUpdate::Fused);
    }

    /// The selection of the original `round()` rule.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a finite pixel inside 0..size"
    )]
    fn rounded(x: f32, size: usize) -> Option<usize> {
        let r = x.round();
        (r.is_finite() && r >= 0.0 && r < size as f32).then_some(r as usize)
    }

    #[test]
    fn nearest_pixel_matches_round_at_every_half_pixel_boundary() {
        for k in -4..5_000 {
            let half = k as f32 + 0.5;
            let mut x = half.next_down().next_down();
            for _ in 0..5 {
                assert_eq!(nearest_pixel(x, 4_096), rounded(x, 4_096), "x = {x:e}");
                x = x.next_up();
            }
        }
        // The largest f32 below 0.5: `x + 0.5` in f32 rounds up to 1.0.
        assert_eq!(nearest_pixel(0.499_999_97, 8), Some(0));
    }

    #[test]
    fn nearest_pixel_matches_round_across_the_f32_range() {
        for bits in (0..=u32::MAX).step_by(9_973) {
            let x = f32::from_bits(bits);
            assert_eq!(nearest_pixel(x, 4_096), rounded(x, 4_096), "x = {x:e}");
        }
        for x in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            4_095.5,
            4_096.0,
        ] {
            assert_eq!(nearest_pixel(x, 4_096), rounded(x, 4_096), "x = {x:e}");
        }
    }
}
