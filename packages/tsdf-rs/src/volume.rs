use std::collections::hash_map::Entry;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::TsdfError;
use crate::marching_cubes::{interpolation_amount, Mesh};
use crate::tables::{EDGE_TABLE, TRI_TABLE};

const BLOCK_RESOLUTION: i64 = 16;
const BLOCK_RESOLUTION_USIZE: usize = BLOCK_RESOLUTION as usize;
const BLOCK_VOXEL_COUNT: usize =
    BLOCK_RESOLUTION_USIZE * BLOCK_RESOLUTION_USIZE * BLOCK_RESOLUTION_USIZE;
const DISCOVERY_CHUNK_PIXELS: usize = 2_048;
const MIN_BLOCK_INDEX: i64 = i64::MIN / BLOCK_RESOLUTION;
const MAX_BLOCK_INDEX: i64 = (i64::MAX - BLOCK_RESOLUTION) / BLOCK_RESOLUTION;

const CORNERS: [[i64; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

const EDGE_GEOMETRY: [[i64; 4]; 12] = [
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 2],
    [1, 0, 0, 2],
    [1, 1, 0, 2],
    [0, 1, 0, 2],
];

type BlockKey = [i64; 3];

#[derive(Clone, Copy)]
struct PixelRay {
    x_over_z: f32,
    y_over_z: f32,
    distance_multiplier: f32,
}

struct RayCache {
    image_size: [usize; 2],
    intrinsics: [f32; 9],
    rays: Vec<PixelRay>,
}

struct VoxelBlock {
    tsdf: Box<[f32]>,
    weights: Box<[f32]>,
    colors: Option<Box<[[f32; 3]]>>,
    color_weights: Option<Box<[f32]>>,
    observed_min: f32,
    observed_max: f32,
}

impl VoxelBlock {
    fn new(fuse_color: bool) -> Self {
        Self {
            tsdf: vec![1.0; BLOCK_VOXEL_COUNT].into_boxed_slice(),
            weights: vec![0.0; BLOCK_VOXEL_COUNT].into_boxed_slice(),
            colors: fuse_color.then(|| vec![[0.0; 3]; BLOCK_VOXEL_COUNT].into_boxed_slice()),
            color_weights: fuse_color.then(|| vec![0.0; BLOCK_VOXEL_COUNT].into_boxed_slice()),
            observed_min: f32::INFINITY,
            observed_max: f32::NEG_INFINITY,
        }
    }
}

/// Sparse, unbounded truncated signed-distance volume.
///
/// The volume allocates 16-by-16-by-16 voxel blocks around observed depth
/// surfaces. Depth values and camera translations must use the same world
/// units as `voxel_size` and `sdf_trunc`.
///
/// # Example
///
/// ```
/// use tsdf_rs::TsdfVolume;
///
/// let volume = TsdfVolume::new(0.04, 0.12, false)?;
/// assert!(volume.extract_mesh()?.vertices.is_empty());
/// # Ok::<(), tsdf_rs::TsdfError>(())
/// ```
pub struct TsdfVolume {
    voxel_size: f32,
    sdf_trunc: f32,
    fuse_color: bool,
    blocks: FxHashMap<BlockKey, VoxelBlock>,
    active_blocks: Vec<(BlockKey, VoxelBlock)>,
    dirty_blocks: FxHashSet<BlockKey>,
    surfaced_blocks: FxHashSet<BlockKey>,
    ray_cache: Option<RayCache>,
}

impl TsdfVolume {
    /// Create an empty sparse TSDF volume.
    ///
    /// Voxels use center samples. Integration truncates signed distances to
    /// `[-sdf_trunc, sdf_trunc]` and accumulates observation weights as `f32`.
    ///
    /// # Arguments
    ///
    /// * `voxel_size` - Edge length of one cubic voxel in world units.
    /// * `sdf_trunc` - Signed-distance truncation in the same units.
    /// * `fuse_color` - Whether to allocate and accumulate RGB values.
    ///
    /// # Returns
    ///
    /// An empty volume ready for depth integration.
    ///
    /// # Errors
    ///
    /// Returns [`TsdfError::InvalidParameter`] if either distance is not finite
    /// and positive.
    ///
    /// # Example
    ///
    /// ```
    /// use tsdf_rs::TsdfVolume;
    ///
    /// let volume = TsdfVolume::new(0.02, 0.06, true)?;
    /// assert!(volume.extract_mesh()?.triangles.is_empty());
    /// # Ok::<(), tsdf_rs::TsdfError>(())
    /// ```
    pub fn new(voxel_size: f32, sdf_trunc: f32, fuse_color: bool) -> Result<Self, TsdfError> {
        validate_positive("voxel_size", voxel_size)?;
        validate_positive("sdf_trunc", sdf_trunc)?;
        Ok(Self {
            voxel_size,
            sdf_trunc,
            fuse_color,
            blocks: FxHashMap::default(),
            active_blocks: Vec::new(),
            dirty_blocks: FxHashSet::default(),
            surfaced_blocks: FxHashSet::default(),
            ray_cache: None,
        })
    }

    /// Fuse one projective depth frame into the sparse volume.
    ///
    /// `camera_from_world` is a row-major rigid world-to-camera transform.
    /// Depth is the camera Z coordinate, not Euclidean ray length. Invalid,
    /// non-finite, and non-positive depth pixels are ignored. Reusing the same
    /// image size and intrinsics reuses a precomputed camera-ray buffer.
    ///
    /// # Arguments
    ///
    /// * `depth` - C-contiguous row-major depth samples.
    /// * `image_size` - Depth dimensions `[height, width]`.
    /// * `intrinsics` - Row-major 3-by-3 camera intrinsic matrix.
    /// * `camera_from_world` - Row-major 4-by-4 world-to-camera transform.
    /// * `color` - Optional row-major RGB bytes with shape `[height, width, 3]`.
    ///
    /// # Returns
    ///
    /// `Ok(())` after all visible voxels have been updated.
    ///
    /// # Errors
    ///
    /// Returns [`TsdfError::ShapeMismatch`] when an input slice length does not
    /// match `image_size`, [`TsdfError::DimensionOverflow`] when a dimension
    /// product overflows, or [`TsdfError::SingularMatrix`] when `intrinsics`
    /// cannot be inverted.
    ///
    /// # Example
    ///
    /// ```
    /// use tsdf_rs::TsdfVolume;
    ///
    /// let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    /// let depth = vec![1.0_f32; 4];
    /// let intrinsics = [100.0, 0.0, 0.5, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0];
    /// let transform = [
    ///     1.0, 0.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// ];
    /// volume.integrate(&depth, [2, 2], intrinsics, transform, None)?;
    /// # Ok::<(), tsdf_rs::TsdfError>(())
    /// ```
    pub fn integrate(
        &mut self,
        depth: &[f32],
        image_size: [usize; 2],
        intrinsics: [f32; 9],
        camera_from_world: [f32; 16],
        color: Option<&[u8]>,
    ) -> Result<(), TsdfError> {
        let [height, width] = image_size;
        let expected_depth = checked_element_count("depth", &image_size)?;
        if depth.len() != expected_depth {
            return Err(TsdfError::ShapeMismatch {
                name: "depth",
                expected: image_size.to_vec(),
                got: vec![depth.len()],
            });
        }
        if let Some(color_values) = color {
            let color_dimensions = [height, width, 3];
            let expected_color = checked_element_count("color", &color_dimensions)?;
            if color_values.len() != expected_color {
                return Err(TsdfError::ShapeMismatch {
                    name: "color",
                    expected: color_dimensions.to_vec(),
                    got: vec![color_values.len()],
                });
            }
        }

        self.update_ray_cache(image_size, intrinsics)?;
        let touched = {
            let rays = self
                .ray_cache
                .as_ref()
                .map(|cache| cache.rays.as_slice())
                .ok_or(TsdfError::SingularMatrix { name: "intrinsics" })?;
            discover_touched_blocks(
                depth,
                rays,
                camera_from_world,
                self.voxel_size,
                self.sdf_trunc,
            )
        };
        if touched.is_empty() {
            return Ok(());
        }

        // A block's mesh reads its {0,1}^3 neighborhood, so updating a block
        // stales the meshes of itself and its negative-side neighbors.
        for &key in &touched {
            for offset in CORNERS {
                let neighbor = [
                    key[0].checked_sub(offset[0]),
                    key[1].checked_sub(offset[1]),
                    key[2].checked_sub(offset[2]),
                ];
                if let [Some(x), Some(y), Some(z)] = neighbor {
                    self.dirty_blocks.insert([x, y, z]);
                }
            }
        }

        for key in touched {
            let block = match self.blocks.remove(&key) {
                Some(block) => block,
                None => VoxelBlock::new(self.fuse_color),
            };
            self.active_blocks.push((key, block));
        }

        let rays = self
            .ray_cache
            .as_ref()
            .map(|cache| cache.rays.as_slice())
            .ok_or(TsdfError::SingularMatrix { name: "intrinsics" })?;
        let frame = Frame {
            depth,
            color,
            rays,
            width,
            height,
            intrinsics,
            camera_from_world,
        };
        let voxel_size = self.voxel_size;
        let sdf_trunc = self.sdf_trunc;
        self.active_blocks.par_iter_mut().for_each(|(key, block)| {
            integrate_block(*key, block, voxel_size, sdf_trunc, &frame);
        });
        for (key, block) in self.active_blocks.drain(..) {
            self.blocks.insert(key, block);
        }
        Ok(())
    }

    /// Extract a deterministic triangle mesh from observed zero crossings.
    ///
    /// Blocks are processed in sorted spatial order. Parallel block results are
    /// merged sequentially in that order, so vertex and triangle bytes do not
    /// depend on the Rayon worker count.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments.
    ///
    /// # Returns
    ///
    /// A mesh containing world-space vertices, triangle indices, and optional
    /// RGB colors.
    ///
    /// # Errors
    ///
    /// Returns [`TsdfError::InvalidEdge`] if the marching-cubes lookup table is
    /// invalid, [`TsdfError::MissingVoxel`] if an emitted edge lacks an
    /// allocated endpoint, or [`TsdfError::VertexIndexOverflow`] if the mesh
    /// exceeds 32-bit indices.
    ///
    /// # Example
    ///
    /// ```
    /// use tsdf_rs::TsdfVolume;
    ///
    /// let volume = TsdfVolume::new(0.04, 0.12, false)?;
    /// let mesh = volume.extract_mesh()?;
    /// assert_eq!(mesh.colors, None);
    /// # Ok::<(), tsdf_rs::TsdfError>(())
    /// ```
    pub fn extract_mesh(&self) -> Result<Mesh, TsdfError> {
        extract_sparse_mesh(&self.blocks, self.voxel_size, self.fuse_color)
    }

    /// Extract per-block meshes for blocks whose surface changed since the
    /// last call, then clear the change tracker.
    ///
    /// Each returned mesh contains only the triangles anchored in that block,
    /// with vertices deduplicated within the block. The union of the latest
    /// mesh per key equals [`TsdfVolume::extract_mesh`] triangle for triangle
    /// (seam vertices are duplicated across blocks). A block whose surface
    /// vanished since it last produced triangles is returned once with an
    /// empty mesh so consumers can clear it; blocks that never had surface
    /// are omitted. Blocks are processed in sorted spatial order, so results
    /// do not depend on the Rayon worker count.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments.
    ///
    /// # Returns
    ///
    /// Sorted `(block_key, mesh)` pairs: every changed block that has surface,
    /// plus one empty mesh per surfaced-to-empty transition. All dirty flags
    /// drain on success.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`TsdfVolume::extract_mesh`]. The change
    /// tracker is only cleared on success.
    ///
    /// # Example
    ///
    /// ```
    /// use tsdf_rs::TsdfVolume;
    ///
    /// let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    /// volume.integrate(
    ///     &[1.0; 4],
    ///     [2, 2],
    ///     [100.0, 0.0, 0.5, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0],
    ///     [
    ///         1.0, 0.0, 0.0, 0.0,
    ///         0.0, 1.0, 0.0, 0.0,
    ///         0.0, 0.0, 1.0, 0.0,
    ///         0.0, 0.0, 0.0, 1.0,
    ///     ],
    ///     None,
    /// )?;
    /// let changed = volume.extract_dirty_block_meshes()?;
    /// assert!(volume.extract_dirty_block_meshes()?.is_empty());
    /// # let _ = changed;
    /// # Ok::<(), tsdf_rs::TsdfError>(())
    /// ```
    pub fn extract_dirty_block_meshes(&mut self) -> Result<Vec<(BlockKey, Mesh)>, TsdfError> {
        let reported = self.dirty_block_meshes_uncommitted()?;
        let outcomes: Vec<(BlockKey, bool)> = reported
            .iter()
            .map(|(key, mesh)| (*key, !mesh.triangles.is_empty()))
            .collect();
        self.commit_dirty_extraction(&outcomes);
        Ok(reported)
    }

    /// Extract dirty-block meshes without touching the change tracker, so a
    /// caller with fallible post-processing (the PyO3 conversion) can commit
    /// only after that work succeeds.
    pub(crate) fn dirty_block_meshes_uncommitted(
        &self,
    ) -> Result<Vec<(BlockKey, Mesh)>, TsdfError> {
        let mut keys: Vec<BlockKey> = self
            .dirty_blocks
            .iter()
            .copied()
            .filter(|key| self.blocks.contains_key(key))
            .collect();
        keys.sort_unstable();

        let meshes: Result<Vec<(BlockKey, Mesh)>, TsdfError> = keys
            .par_iter()
            .map(|&key| {
                let triangles = extract_block_triangles(key, &self.blocks)?;
                let triangle_count = triangles.len();
                let mesh = mesh_from_edge_triangles(
                    triangles,
                    triangle_count,
                    &self.blocks,
                    self.voxel_size,
                    self.fuse_color,
                )?;
                Ok((key, mesh))
            })
            .collect();
        // Report empty meshes only on the surfaced -> empty transition, so
        // surface-less shell blocks are not re-announced every extraction.
        Ok(meshes?
            .into_iter()
            .filter(|(key, mesh)| !mesh.triangles.is_empty() || self.surfaced_blocks.contains(key))
            .collect())
    }

    /// Record extraction outcomes as `(block_key, has_surface)` pairs and
    /// drain the change tracker.
    pub(crate) fn commit_dirty_extraction(&mut self, outcomes: &[(BlockKey, bool)]) {
        for &(key, has_surface) in outcomes {
            if has_surface {
                self.surfaced_blocks.insert(key);
            } else {
                self.surfaced_blocks.remove(&key);
            }
        }
        self.dirty_blocks.clear();
    }

    fn update_ray_cache(
        &mut self,
        image_size: [usize; 2],
        intrinsics: [f32; 9],
    ) -> Result<(), TsdfError> {
        let cache_matches = self
            .ray_cache
            .as_ref()
            .is_some_and(|cache| cache.image_size == image_size && cache.intrinsics == intrinsics);
        if cache_matches {
            return Ok(());
        }

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
        self.ray_cache = Some(RayCache {
            image_size,
            intrinsics,
            rays,
        });
        Ok(())
    }
}

struct Frame<'a> {
    depth: &'a [f32],
    color: Option<&'a [u8]>,
    rays: &'a [PixelRay],
    width: usize,
    height: usize,
    intrinsics: [f32; 9],
    camera_from_world: [f32; 16],
}

fn discover_touched_blocks(
    depth: &[f32],
    rays: &[PixelRay],
    camera_from_world: [f32; 16],
    voxel_size: f32,
    sdf_trunc: f32,
) -> FxHashSet<BlockKey> {
    let block_length = f64::from(voxel_size) * BLOCK_RESOLUTION as f64;
    let sdf_trunc = f64::from(sdf_trunc);
    depth
        .par_chunks(DISCOVERY_CHUNK_PIXELS)
        .zip(rays.par_chunks(DISCOVERY_CHUNK_PIXELS))
        .map(|(depth_chunk, ray_chunk)| {
            let mut local = FxHashSet::default();
            local.reserve(32);
            for (&depth_value, ray) in depth_chunk.iter().zip(ray_chunk) {
                if depth_value <= 0.0
                    || !depth_value.is_finite()
                    || !ray.x_over_z.is_finite()
                    || !ray.y_over_z.is_finite()
                {
                    continue;
                }
                let camera_point = [
                    ray.x_over_z * depth_value,
                    ray.y_over_z * depth_value,
                    depth_value,
                ];
                let world_point = camera_to_world(camera_from_world, camera_point);
                if !world_point.iter().all(|coordinate| coordinate.is_finite()) {
                    continue;
                }

                let Some((min, max)) = block_bounds(world_point, sdf_trunc, block_length) else {
                    continue;
                };
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
    let minimum = [
        checked_block_index((f64::from(world_point[0]) - sdf_trunc) / block_length)?,
        checked_block_index((f64::from(world_point[1]) - sdf_trunc) / block_length)?,
        checked_block_index((f64::from(world_point[2]) - sdf_trunc) / block_length)?,
    ];
    let maximum = [
        checked_block_index((f64::from(world_point[0]) + sdf_trunc) / block_length)?,
        checked_block_index((f64::from(world_point[1]) + sdf_trunc) / block_length)?,
        checked_block_index((f64::from(world_point[2]) + sdf_trunc) / block_length)?,
    ];
    Some((minimum, maximum))
}

fn checked_block_index(value: f64) -> Option<i64> {
    let index = checked_floor_to_i64(value)?;
    (MIN_BLOCK_INDEX..=MAX_BLOCK_INDEX)
        .contains(&index)
        .then_some(index)
}

fn checked_floor_to_i64(value: f64) -> Option<i64> {
    let value = value.floor();
    let upper_bound = -(i64::MIN as f64);
    if value.is_finite() && value >= i64::MIN as f64 && value < upper_bound {
        Some(value as i64)
    } else {
        None
    }
}

fn checked_block_origin(key: BlockKey) -> Option<[i64; 3]> {
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

fn integrate_block(
    key: BlockKey,
    block: &mut VoxelBlock,
    voxel_size: f32,
    sdf_trunc: f32,
    frame: &Frame<'_>,
) {
    let Some(block_origin) = checked_block_origin(key) else {
        return;
    };
    let first_world = block_origin.map(|coordinate| (coordinate as f32 + 0.5) * voxel_size);
    let first_camera = transform_point(frame.camera_from_world, first_world);
    let step_x = [
        frame.camera_from_world[0] * voxel_size,
        frame.camera_from_world[4] * voxel_size,
        frame.camera_from_world[8] * voxel_size,
    ];
    let step_y = [
        frame.camera_from_world[1] * voxel_size,
        frame.camera_from_world[5] * voxel_size,
        frame.camera_from_world[9] * voxel_size,
    ];
    let step_z = [
        frame.camera_from_world[2] * voxel_size,
        frame.camera_from_world[6] * voxel_size,
        frame.camera_from_world[10] * voxel_size,
    ];

    for z in 0..BLOCK_RESOLUTION_USIZE {
        let camera_at_z = add_scaled(first_camera, step_z, z as f32);
        for y in 0..BLOCK_RESOLUTION_USIZE {
            let mut camera = add_scaled(camera_at_z, step_y, y as f32);
            for x in 0..BLOCK_RESOLUTION_USIZE {
                let index = (z * BLOCK_RESOLUTION_USIZE + y) * BLOCK_RESOLUTION_USIZE + x;
                integrate_voxel(index, camera, block, sdf_trunc, frame);
                camera = add(camera, step_x);
            }
        }
    }
}

fn integrate_voxel(
    index: usize,
    camera: [f32; 3],
    block: &mut VoxelBlock,
    sdf_trunc: f32,
    frame: &Frame<'_>,
) {
    if camera[2] <= 0.0 || !camera[2].is_finite() {
        return;
    }
    let pixel = multiply_3x3(frame.intrinsics, camera);
    if pixel[2] == 0.0 || !pixel[2].is_finite() {
        return;
    }
    let u = (pixel[0] / pixel[2]).round();
    let v = (pixel[1] / pixel[2]).round();
    if !u.is_finite()
        || !v.is_finite()
        || u < 0.0
        || v < 0.0
        || u >= frame.width as f32
        || v >= frame.height as f32
    {
        return;
    }

    let pixel_index = v as usize * frame.width + u as usize;
    let depth = frame.depth[pixel_index];
    if depth <= 0.0 || !depth.is_finite() {
        return;
    }
    let distance_multiplier = frame.rays[pixel_index].distance_multiplier;
    if !distance_multiplier.is_finite() {
        return;
    }
    let signed_distance = (depth - camera[2]) * distance_multiplier;
    if signed_distance < -sdf_trunc {
        return;
    }

    let observation = (signed_distance / sdf_trunc).min(1.0);
    let old_weight = block.weights[index];
    let denominator = old_weight + 1.0;
    let updated_tsdf = (block.tsdf[index] * old_weight + observation) / denominator;
    block.tsdf[index] = updated_tsdf;
    block.weights[index] = denominator;
    block.observed_min = block.observed_min.min(updated_tsdf);
    block.observed_max = block.observed_max.max(updated_tsdf);

    if let (Some(image), Some(colors), Some(color_weights)) = (
        frame.color,
        block.colors.as_mut(),
        block.color_weights.as_mut(),
    ) {
        let old_color_weight = color_weights[index];
        let color_denominator = old_color_weight + 1.0;
        let image_offset = pixel_index * 3;
        for channel in 0..3 {
            colors[index][channel] = (colors[index][channel] * old_color_weight
                + image[image_offset + channel] as f32)
                / color_denominator;
        }
        color_weights[index] = color_denominator;
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct EdgeKey {
    base: [i64; 3],
    axis: u8,
}

fn extract_sparse_mesh(
    blocks: &FxHashMap<BlockKey, VoxelBlock>,
    voxel_size: f32,
    fuse_color: bool,
) -> Result<Mesh, TsdfError> {
    let mut block_keys: Vec<_> = blocks.keys().copied().collect();
    block_keys.sort_unstable();

    let block_triangles: Result<Vec<Vec<[EdgeKey; 3]>>, TsdfError> = block_keys
        .par_iter()
        .map(|&key| extract_block_triangles(key, blocks))
        .collect();
    let block_triangles = block_triangles?;
    let triangle_count = block_triangles.iter().map(Vec::len).sum();
    mesh_from_edge_triangles(
        block_triangles.into_iter().flatten(),
        triangle_count,
        blocks,
        voxel_size,
        fuse_color,
    )
}

/// Deduplicate edge-keyed triangles into an indexed mesh with interpolated
/// vertex positions and optional nearest-voxel colors.
fn mesh_from_edge_triangles(
    edge_triangles: impl IntoIterator<Item = [EdgeKey; 3]>,
    triangle_count: usize,
    blocks: &FxHashMap<BlockKey, VoxelBlock>,
    voxel_size: f32,
    fuse_color: bool,
) -> Result<Mesh, TsdfError> {
    let mut edge_to_vertex = FxHashMap::<EdgeKey, u32>::default();
    edge_to_vertex.reserve(triangle_count);
    let mut vertices = Vec::with_capacity(triangle_count / 2);
    let mut triangles = Vec::with_capacity(triangle_count);

    for edge_triangle in edge_triangles {
        let mut triangle = [0_u32; 3];
        for (destination, edge) in triangle.iter_mut().zip(edge_triangle) {
            *destination = match edge_to_vertex.entry(edge) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let vertex_index = u32::try_from(vertices.len()).map_err(|_| {
                        TsdfError::VertexIndexOverflow {
                            vertex_count: vertices.len(),
                        }
                    })?;
                    vertices.push(interpolate_sparse_edge(edge, blocks, voxel_size)?);
                    entry.insert(vertex_index);
                    vertex_index
                }
            };
        }
        triangles.push(triangle);
    }

    let colors = fuse_color.then(|| {
        vertices
            .iter()
            .map(|vertex| nearest_color(vertex, blocks, voxel_size))
            .collect()
    });
    Ok(Mesh {
        vertices,
        triangles,
        colors,
    })
}

fn extract_block_triangles(
    key: BlockKey,
    blocks: &FxHashMap<BlockKey, VoxelBlock>,
) -> Result<Vec<[EdgeKey; 3]>, TsdfError> {
    let neighbors = neighbor_blocks(key, blocks);
    if !has_possible_surface(&neighbors) {
        return Ok(Vec::new());
    }

    let Some(block_origin) = checked_block_origin(key) else {
        return Ok(Vec::new());
    };
    let mut triangles = Vec::with_capacity(512);
    for z in 0..BLOCK_RESOLUTION {
        for y in 0..BLOCK_RESOLUTION {
            for x in 0..BLOCK_RESOLUTION {
                let anchor = [
                    block_origin[0] + x,
                    block_origin[1] + y,
                    block_origin[2] + z,
                ];
                let mut cube_index = 0usize;
                let mut observed = true;
                for (corner, offset) in CORNERS.iter().enumerate() {
                    let local = [x + offset[0], y + offset[1], z + offset[2]];
                    let Some((value, weight)) = cached_voxel_value(&neighbors, local) else {
                        observed = false;
                        break;
                    };
                    if weight == 0.0 || !value.is_finite() {
                        observed = false;
                        break;
                    }
                    if value < 0.0 {
                        cube_index |= 1 << corner;
                    }
                }
                if !observed || EDGE_TABLE[cube_index] == 0 {
                    continue;
                }

                let row = &TRI_TABLE[cube_index];
                for triangle_edges in row.chunks_exact(3) {
                    if triangle_edges[0] < 0 {
                        break;
                    }
                    // Emit vertices 0,2,1 to match Open3D's counter-clockwise
                    // winding (see the same flip in marching_cubes.rs).
                    triangles.push([
                        edge_key(anchor, triangle_edges[0] as usize)?,
                        edge_key(anchor, triangle_edges[2] as usize)?,
                        edge_key(anchor, triangle_edges[1] as usize)?,
                    ]);
                }
            }
        }
    }
    Ok(triangles)
}

fn neighbor_blocks(
    key: BlockKey,
    blocks: &FxHashMap<BlockKey, VoxelBlock>,
) -> [Option<&VoxelBlock>; 8] {
    std::array::from_fn(|index| {
        let neighbor_key = [
            key[0].checked_add((index & 1) as i64)?,
            key[1].checked_add(((index >> 1) & 1) as i64)?,
            key[2].checked_add(((index >> 2) & 1) as i64)?,
        ];
        blocks.get(&neighbor_key)
    })
}

fn has_possible_surface(neighbors: &[Option<&VoxelBlock>; 8]) -> bool {
    let (minimum, maximum) = neighbors.iter().flatten().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), block| {
            (
                minimum.min(block.observed_min),
                maximum.max(block.observed_max),
            )
        },
    );
    minimum < 0.0 && maximum >= 0.0
}

fn cached_voxel_value(neighbors: &[Option<&VoxelBlock>; 8], local: [i64; 3]) -> Option<(f32, f32)> {
    let neighbor_index = (local[0] / BLOCK_RESOLUTION)
        | ((local[1] / BLOCK_RESOLUTION) << 1)
        | ((local[2] / BLOCK_RESOLUTION) << 2);
    let block = neighbors[neighbor_index as usize]?;
    let wrapped = local.map(|coordinate| coordinate % BLOCK_RESOLUTION);
    let index = local_index(wrapped);
    Some((block.tsdf[index], block.weights[index]))
}

fn edge_key(anchor: [i64; 3], edge: usize) -> Result<EdgeKey, TsdfError> {
    let geometry = EDGE_GEOMETRY
        .get(edge)
        .ok_or(TsdfError::InvalidEdge { edge })?;
    Ok(EdgeKey {
        base: [
            anchor[0] + geometry[0],
            anchor[1] + geometry[1],
            anchor[2] + geometry[2],
        ],
        axis: geometry[3] as u8,
    })
}

fn interpolate_sparse_edge(
    edge: EdgeKey,
    blocks: &FxHashMap<BlockKey, VoxelBlock>,
    voxel_size: f32,
) -> Result<[f32; 3], TsdfError> {
    let mut other = edge.base;
    other[edge.axis as usize] += 1;
    let value_a = voxel_value(blocks, edge.base)
        .ok_or(TsdfError::MissingVoxel {
            coordinate: edge.base,
        })?
        .0;
    let value_b = voxel_value(blocks, other)
        .ok_or(TsdfError::MissingVoxel { coordinate: other })?
        .0;
    let amount = interpolation_amount(value_a, value_b);

    let mut position = edge
        .base
        .map(|coordinate| (coordinate as f32 + 0.5) * voxel_size);
    position[edge.axis as usize] += amount * voxel_size;
    Ok(position)
}

fn voxel_value(
    blocks: &FxHashMap<BlockKey, VoxelBlock>,
    coordinate: [i64; 3],
) -> Option<(f32, f32)> {
    let (key, index) = voxel_location(coordinate);
    blocks
        .get(&key)
        .map(|block| (block.tsdf[index], block.weights[index]))
}

fn voxel_location(coordinate: [i64; 3]) -> (BlockKey, usize) {
    let key = coordinate.map(|value| value.div_euclid(BLOCK_RESOLUTION));
    let local = coordinate.map(|value| value.rem_euclid(BLOCK_RESOLUTION));
    (key, local_index(local))
}

fn local_index(local: [i64; 3]) -> usize {
    let [x, y, z] = local.map(|coordinate| coordinate as usize);
    (z * BLOCK_RESOLUTION_USIZE + y) * BLOCK_RESOLUTION_USIZE + x
}

fn nearest_color(
    vertex: &[f32; 3],
    blocks: &FxHashMap<BlockKey, VoxelBlock>,
    voxel_size: f32,
) -> [u8; 3] {
    let Some(coordinate) = world_to_voxel_coordinate(*vertex, voxel_size) else {
        return [0; 3];
    };
    let (key, index) = voxel_location(coordinate);
    blocks
        .get(&key)
        .and_then(|block| block.colors.as_ref())
        .map_or([0; 3], |colors| {
            colors[index].map(|value| value.round().clamp(0.0, 255.0) as u8)
        })
}

fn world_to_voxel_coordinate(vertex: [f32; 3], voxel_size: f32) -> Option<[i64; 3]> {
    let voxel_size = f64::from(voxel_size);
    Some([
        checked_round_to_i64(f64::from(vertex[0]) / voxel_size - 0.5)?,
        checked_round_to_i64(f64::from(vertex[1]) / voxel_size - 0.5)?,
        checked_round_to_i64(f64::from(vertex[2]) / voxel_size - 0.5)?,
    ])
}

fn checked_round_to_i64(value: f64) -> Option<i64> {
    let value = value.round();
    let upper_bound = -(i64::MIN as f64);
    if value.is_finite() && value >= i64::MIN as f64 && value < upper_bound {
        Some(value as i64)
    } else {
        None
    }
}

fn transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
    ]
}

fn camera_to_world(camera_from_world: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    let shifted = [
        point[0] - camera_from_world[3],
        point[1] - camera_from_world[7],
        point[2] - camera_from_world[11],
    ];
    [
        camera_from_world[0] * shifted[0]
            + camera_from_world[4] * shifted[1]
            + camera_from_world[8] * shifted[2],
        camera_from_world[1] * shifted[0]
            + camera_from_world[5] * shifted[1]
            + camera_from_world[9] * shifted[2],
        camera_from_world[2] * shifted[0]
            + camera_from_world[6] * shifted[1]
            + camera_from_world[10] * shifted[2],
    ]
}

fn multiply_3x3(matrix: [f32; 9], vector: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    ]
}

fn invert_3x3(matrix: [f32; 9]) -> Option<[f32; 9]> {
    let determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    Some([
        (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inverse_determinant,
        (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse_determinant,
        (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse_determinant,
        (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inverse_determinant,
        (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse_determinant,
        (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse_determinant,
        (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inverse_determinant,
        (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inverse_determinant,
        (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse_determinant,
    ])
}

fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn add_scaled(base: [f32; 3], step: [f32; 3], amount: f32) -> [f32; 3] {
    [
        base[0] + step[0] * amount,
        base[1] + step[1] * amount,
        base[2] + step[2] * amount,
    ]
}

fn validate_positive(name: &'static str, value: f32) -> Result<(), TsdfError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(TsdfError::InvalidParameter {
            name,
            reason: "must be finite and positive",
            value,
        })
    }
}

fn checked_element_count(name: &'static str, dimensions: &[usize]) -> Result<usize, TsdfError> {
    dimensions.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| TsdfError::DimensionOverflow {
                name,
                dimensions: dimensions.to_vec(),
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_cube_with_a_nan_sample_is_unobserved() -> Result<(), TsdfError> {
        let mut block = VoxelBlock::new(false);
        for offset in CORNERS {
            block.weights[local_index(offset)] = 1.0;
        }
        block.tsdf[local_index(CORNERS[0])] = -1.0;
        block.tsdf[local_index(CORNERS[1])] = f32::NAN;
        block.observed_min = -1.0;
        block.observed_max = 1.0;

        let mut blocks = FxHashMap::default();
        blocks.insert([0, 0, 0], block);
        let mesh = extract_sparse_mesh(&blocks, 0.04, false)?;

        assert!(mesh.vertices.is_empty());
        assert!(mesh.triangles.is_empty());
        Ok(())
    }
}
