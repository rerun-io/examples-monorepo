use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::changes::{ChangeTracker, DirtyMeshes};
use crate::error::TsdfError;
use crate::extract;
use crate::frame::{Camera, DepthFrame, DepthRange};
use crate::geometry::{
    add, add_scaled, camera_to_world, checked_floor_i64, invert_3x3, multiply_3x3, transform_point,
    validate_length, validate_positive,
};
use crate::mesh::Mesh;

pub(crate) const BLOCK_RESOLUTION: i64 = 16;
const BLOCK_RESOLUTION_USIZE: usize = BLOCK_RESOLUTION as usize;
const BLOCK_VOXEL_COUNT: usize =
    BLOCK_RESOLUTION_USIZE * BLOCK_RESOLUTION_USIZE * BLOCK_RESOLUTION_USIZE;
const DISCOVERY_CHUNK_PIXELS: usize = 2_048;
const MIN_BLOCK_INDEX: i64 = i64::MIN / BLOCK_RESOLUTION;
const MAX_BLOCK_INDEX: i64 = (i64::MAX - BLOCK_RESOLUTION) / BLOCK_RESOLUTION;

/// Integer index of a 16-by-16-by-16 voxel block: block `k` holds global voxel
/// coordinates `16 * k` to `16 * k + 15` on each axis.
pub type BlockKey = [i64; 3];

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
                    distance_multiplier: (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2])
                        .sqrt()
                        / ray[2].abs(),
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

/// Index of block-local voxel `local` (each coordinate in `0..16`).
pub(crate) fn local_index(local: [i64; 3]) -> usize {
    let [x, y, z] = local.map(|coordinate| coordinate as usize);
    (z * BLOCK_RESOLUTION_USIZE + y) * BLOCK_RESOLUTION_USIZE + x
}

/// Global voxel coordinate of the first voxel of block `key`, when every
/// voxel of the block has an `i64` coordinate.
pub(crate) fn checked_block_origin(key: BlockKey) -> Option<[i64; 3]> {
    let origin = [
        key[0].checked_mul(BLOCK_RESOLUTION)?,
        key[1].checked_mul(BLOCK_RESOLUTION)?,
        key[2].checked_mul(BLOCK_RESOLUTION)?,
    ];
    origin[0].checked_add(BLOCK_RESOLUTION)?;
    origin[1].checked_add(BLOCK_RESOLUTION)?;
    origin[2].checked_add(BLOCK_RESOLUTION)?;
    Some(origin)
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
    blocks: FxHashMap<BlockKey, VoxelBlock>,
    changes: ChangeTracker,
    ray_cache: Option<RayCache>,
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
            blocks: FxHashMap::default(),
            changes: ChangeTracker::default(),
            ray_cache: None,
        })
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
        };
        let touched = discover_touched_blocks(&context);
        self.changes.mark_touched(&touched);

        let mut active: Vec<(BlockKey, VoxelBlock)> = touched
            .into_iter()
            .map(|key| {
                let block = self
                    .blocks
                    .remove(&key)
                    .unwrap_or_else(|| VoxelBlock::new(self.fuse_color));
                (key, block)
            })
            .collect();
        active
            .par_iter_mut()
            .for_each(|(key, block)| integrate_block(*key, block, &context));
        self.blocks.extend(active);
        Ok(())
    }

    /// Extract the mesh of every observed zero crossing.
    ///
    /// Vertices lie on cell edges whose eight-corner cell is fully observed;
    /// each carries the color of its nearest voxel and the unit TSDF gradient
    /// as normal. Blocks are processed in sorted order and merged
    /// sequentially, so the bytes do not depend on the Rayon thread count.
    ///
    /// # Errors
    ///
    /// [`TsdfError::VertexIndexOverflow`] if the mesh exceeds 32-bit indices.
    pub fn extract_mesh(&self) -> Result<Mesh, TsdfError> {
        extract::full_mesh(&self.blocks, self.voxel_size, self.fuse_color)
    }

    /// Per-block meshes of the blocks whose surface may have changed since
    /// the last confirmed [`crate::Delivery`].
    ///
    /// Each mesh holds the triangles of cells anchored in its block, with
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
        let meshes = extract::block_meshes(
            &keys,
            &self.blocks,
            &self.changes.surfaced,
            self.voxel_size,
            self.fuse_color,
        );
        self.changes.pending(meshes)
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
}

/// Blocks within `sdf_trunc` of a back-projected valid depth sample.
fn discover_touched_blocks(context: &FrameContext<'_>) -> FxHashSet<BlockKey> {
    let block_length = f64::from(context.voxel_size) * BLOCK_RESOLUTION as f64;
    let sdf_trunc = f64::from(context.sdf_trunc);
    context
        .depth
        .par_chunks(DISCOVERY_CHUNK_PIXELS)
        .zip(context.rays.par_chunks(DISCOVERY_CHUNK_PIXELS))
        .map(|(depth_chunk, ray_chunk)| {
            let mut local = FxHashSet::default();
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
                let Some(bounds) = block_bounds(world_point, sdf_trunc, block_length) else {
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

fn block_bounds(
    world_point: [f32; 3],
    sdf_trunc: f64,
    block_length: f64,
) -> Option<(BlockKey, BlockKey)> {
    let block_index = |offset: f64| -> Option<BlockKey> {
        let mut key = [0; 3];
        for (index, &coordinate) in key.iter_mut().zip(&world_point) {
            *index = checked_floor_i64((f64::from(coordinate) + offset) / block_length)
                .filter(|index| (MIN_BLOCK_INDEX..=MAX_BLOCK_INDEX).contains(index))?;
        }
        Some(key)
    };
    Some((block_index(-sdf_trunc)?, block_index(sdf_trunc)?))
}

fn integrate_block(key: BlockKey, block: &mut VoxelBlock, context: &FrameContext<'_>) {
    let Some(block_origin) = checked_block_origin(key) else {
        return;
    };
    let voxel_size = context.voxel_size;
    let transform = context.camera_from_world;
    let first_world = block_origin.map(|coordinate| (coordinate as f32 + 0.5) * voxel_size);
    let first_camera = transform_point(transform, first_world);
    let step_x = [
        transform[0] * voxel_size,
        transform[4] * voxel_size,
        transform[8] * voxel_size,
    ];
    let step_y = [
        transform[1] * voxel_size,
        transform[5] * voxel_size,
        transform[9] * voxel_size,
    ];
    let step_z = [
        transform[2] * voxel_size,
        transform[6] * voxel_size,
        transform[10] * voxel_size,
    ];

    for z in 0..BLOCK_RESOLUTION_USIZE {
        let camera_at_z = add_scaled(first_camera, step_z, z as f32);
        for y in 0..BLOCK_RESOLUTION_USIZE {
            let mut camera = add_scaled(camera_at_z, step_y, y as f32);
            for x in 0..BLOCK_RESOLUTION_USIZE {
                let index = (z * BLOCK_RESOLUTION_USIZE + y) * BLOCK_RESOLUTION_USIZE + x;
                integrate_voxel(index, camera, block, context);
                camera = add(camera, step_x);
            }
        }
    }
}

fn integrate_voxel(
    index: usize,
    camera: [f32; 3],
    block: &mut VoxelBlock,
    context: &FrameContext<'_>,
) {
    if camera[2] <= 0.0 || !camera[2].is_finite() {
        return;
    }
    let pixel = multiply_3x3(context.intrinsics, camera);
    if pixel[2] == 0.0 || !pixel[2].is_finite() {
        return;
    }
    let (Some(u), Some(v)) = (
        nearest_pixel(pixel[0] / pixel[2], context.width),
        nearest_pixel(pixel[1] / pixel[2], context.height),
    ) else {
        return;
    };

    let pixel_index = v * context.width + u;
    let depth = context.depth[pixel_index];
    if !context.depth_range.contains(depth) {
        return;
    }
    let distance_multiplier = context.rays[pixel_index].distance_multiplier;
    if !distance_multiplier.is_finite() {
        return;
    }
    let signed_distance = (depth - camera[2]) * distance_multiplier;
    if signed_distance < -context.sdf_trunc {
        return;
    }

    let observation = (signed_distance / context.sdf_trunc).min(1.0);
    let old_weight = block.weights[index];
    let denominator = old_weight + 1.0;
    let updated_tsdf = (block.tsdf[index] * old_weight + observation) / denominator;
    block.tsdf[index] = updated_tsdf;
    block.weights[index] = denominator;
    block.observed_min = block.observed_min.min(updated_tsdf);
    block.observed_max = block.observed_max.max(updated_tsdf);

    if let (Some(image), Some(color)) = (context.color, block.color.as_mut()) {
        let old_weight = color.weight[index];
        let denominator = old_weight + 1.0;
        let image_offset = pixel_index * 3;
        for channel in 0..3 {
            color.rgb[index][channel] = (color.rgb[index][channel] * old_weight
                + f32::from(image[image_offset + channel]))
                / denominator;
        }
        color.weight[index] = denominator;
    }
}

/// The pixel `x.round()` (half away from zero) selects on an axis of `size`
/// pixels, if inside, without the libm `round` call: in `f64`, `x + 0.5`
/// never rounds across an integer for an `f32` `x`, so truncating it rounds.
#[inline]
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
    use super::nearest_pixel;

    /// The selection of the original `round()` rule.
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
