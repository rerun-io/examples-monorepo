//! Marching-cubes extraction over the sparse blocks: the full mesh and per-block meshes.

use std::collections::hash_map::Entry;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::changes::{ChangeTracker, Pending, NEIGHBORHOOD};
use crate::error::TsdfError;
use crate::geometry::{add, length};
use crate::mesh::Mesh;
use crate::tables::{for_each_triangle, CORNERS, EDGES};
#[cfg(test)]
use crate::volume::MIN_BLOCK_INDEX;
use crate::volume::{
    checked_block_origin, checked_offset, local_index, voxel_center, voxel_index, BlockKey, Blocks,
    VoxelBlock, BLOCK_RESOLUTION, BLOCK_RESOLUTION_USIZE,
};

/// A cell edge: its lower voxel and the axis to the other one.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct EdgeKey {
    base: [i64; 3],
    axis: u8,
}

impl EdgeKey {
    fn end(self) -> [i64; 3] {
        let mut end = self.base;
        end[usize::from(self.axis)] += 1;
        end
    }
}

/// A mesh vertex: the zero crossing on an edge.
#[derive(Clone, Copy)]
struct EdgeVertex {
    edge: EdgeKey,
    /// Fraction of the way from `edge.base` to its end.
    amount: f32,
    /// Whether cells of another block share the edge.
    on_seam: bool,
}

/// One block's triangles with block-local, first-seen vertex indices.
#[derive(Default)]
struct BlockMesh {
    vertices: Vec<EdgeVertex>,
    triangles: Vec<[u32; 3]>,
}

/// What extraction reads of a volume.
#[derive(Clone, Copy)]
pub(crate) struct VolumeView<'a> {
    pub(crate) blocks: &'a Blocks,
    pub(crate) voxel_size: f32,
    pub(crate) fuse_color: bool,
    /// Cells with a corner fused fewer times are left out.
    pub(crate) min_weight: f32,
}

/// Full extraction: blocks mesh in parallel, then a sequential merge in
/// sorted block order joins vertices on block seams. The vertex order is the
/// first-seen order of the whole sorted triangle stream.
pub(crate) fn full_mesh(volume: &VolumeView<'_>) -> Result<Mesh, TsdfError> {
    let VolumeView {
        blocks,
        voxel_size,
        fuse_color,
        min_weight,
    } = *volume;
    let mut keys: Vec<BlockKey> = blocks.keys().copied().collect();
    keys.sort_unstable();
    let block_meshes: Vec<BlockMesh> = keys
        .par_iter()
        .map_init(MeshScratch::new, |scratch, &key| {
            block_mesh(key, blocks, min_weight, scratch)
        })
        .collect();

    let triangle_count = block_meshes.iter().map(|mesh| mesh.triangles.len()).sum();
    let seam_count = block_meshes
        .iter()
        .map(|mesh| mesh.vertices.iter().filter(|vertex| vertex.on_seam).count())
        .sum();
    let mut seams = FxHashMap::<EdgeKey, u32>::default();
    seams.reserve(seam_count);
    let mut vertices: Vec<EdgeVertex> = Vec::with_capacity(triangle_count / 2);
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(triangle_count);
    let mut remap: Vec<u32> = Vec::new();
    for block in block_meshes {
        remap.clear();
        for vertex in block.vertices {
            let next =
                u32::try_from(vertices.len()).map_err(|_| TsdfError::VertexIndexOverflow {
                    vertex_count: vertices.len(),
                })?;
            let index = if vertex.on_seam {
                match seams.entry(vertex.edge) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        vertices.push(vertex);
                        *entry.insert(next)
                    }
                }
            } else {
                vertices.push(vertex);
                next
            };
            remap.push(index);
        }
        triangles.extend(
            block
                .triangles
                .iter()
                .map(|triangle| triangle.map(|local| remap[local as usize])),
        );
    }
    // Whole-volume vertex lists are large: attributes in parallel.
    let positions: Vec<[f32; 3]> = vertices
        .par_iter()
        .map(|vertex| vertex_position(vertex, voxel_size))
        .collect();
    let colors = fuse_color.then(|| {
        positions
            .par_iter()
            .map(|position| nearest_color(*position, blocks, voxel_size))
            .collect()
    });
    let normals = vertices
        .par_iter()
        .map(|vertex| vertex_normal(vertex, blocks))
        .collect();
    Ok(Mesh {
        vertices: positions,
        triangles,
        colors,
        normals,
    })
}

/// Meshes of `keys` (sorted) that differ from the block's last delivered
/// mesh in `changes`.
pub(crate) fn block_meshes(
    volume: &VolumeView<'_>,
    keys: &[BlockKey],
    changes: &ChangeTracker,
) -> Vec<Pending> {
    let blocks = volume.blocks;
    keys.par_iter()
        .map_init(MeshScratch::new, |scratch, &key| {
            let block = block_mesh(key, blocks, volume.min_weight, scratch);
            let last = changes.last_surface_hash(&key);
            // Report empty meshes only on the surfaced -> empty transition, so
            // surface-less shell blocks are not re-announced every extraction.
            if block.triangles.is_empty() && last.is_none() {
                return None;
            }
            // Block jobs already run in parallel: attributes sequentially,
            // through the block's neighborhood.
            let mesh = match Neighborhood::new(key, blocks) {
                Some(voxels) => block_attributes(block, &voxels, volume),
                None => block_attributes(block, blocks, volume),
            };
            // A dirty block's mesh is often unchanged (its neighbor was
            // touched, not it): leave out what the consumer already holds.
            let surface_hash = (!mesh.triangles.is_empty()).then(|| mesh.content_hash());
            (surface_hash != last).then_some(Pending {
                key,
                mesh,
                surface_hash,
            })
        })
        .flatten()
        .collect()
}

/// Positions, colors and normals of a block's vertices, computed in order.
fn block_attributes(block: BlockMesh, voxels: &impl Voxels, volume: &VolumeView<'_>) -> Mesh {
    let VolumeView {
        voxel_size,
        fuse_color,
        ..
    } = *volume;
    let positions: Vec<[f32; 3]> = block
        .vertices
        .iter()
        .map(|vertex| vertex_position(vertex, voxel_size))
        .collect();
    let colors = fuse_color.then(|| {
        positions
            .iter()
            .map(|position| nearest_color(*position, voxels, voxel_size))
            .collect()
    });
    let normals = block
        .vertices
        .iter()
        .map(|vertex| vertex_normal(vertex, voxels))
        .collect();
    Mesh {
        vertices: positions,
        triangles: block.triangles,
        colors,
        normals,
    }
}

/// World position of a vertex: the voxel center of its edge's base, moved
/// `amount` voxels along the edge.
fn vertex_position(vertex: &EdgeVertex, voxel_size: f32) -> [f32; 3] {
    let mut position = voxel_center(vertex.edge.base, voxel_size);
    position[usize::from(vertex.edge.axis)] += vertex.amount * voxel_size;
    position
}

/// [`BLOCK_RESOLUTION`] as a loop counter that converts losslessly to `usize` and `i64`.
const BLOCK_RESOLUTION_U16: u16 = BLOCK_RESOLUTION_USIZE as u16;

/// Voxels per axis that a block's cells read: its own and the first of the next block.
const SPAN: usize = BLOCK_RESOLUTION_USIZE + 1;

/// Offsets of the cell corners, in [`CORNERS`] order, in a [`SPAN`]-cubed
/// array laid out x fastest.
const CORNER_OFFSETS: [usize; 8] = {
    let mut offsets = [0; 8];
    let mut corner = 0;
    while corner < 8 {
        let [x, y, z] = CORNERS[corner];
        offsets[corner] = (z as usize * SPAN + y as usize) * SPAN + x as usize;
        corner += 1;
    }
    offsets
};

/// Offsets of the twelve cell edges, in [`EDGES`] order, among the vertex
/// slots of a block: three per voxel of a [`SPAN`]-cubed array (one per axis).
const EDGE_SLOT_OFFSETS: [usize; 12] = {
    let mut offsets = [0; 12];
    let mut edge = 0;
    while edge < 12 {
        let [x, y, z] = EDGES[edge].offset;
        let voxel = (z as usize * SPAN + y as usize) * SPAN + x as usize;
        offsets[edge] = voxel * 3 + EDGES[edge].axis as usize;
        edge += 1;
    }
    offsets
};

/// Marks a vertex slot whose edge has no vertex yet.
const NO_VERTEX: u32 = u32::MAX;

/// Buffers one worker reuses from block to block.
pub(crate) struct MeshScratch {
    /// The corner values a block's cells read, [`SPAN`]-cubed, x fastest;
    /// NaN where a cell may not use the voxel.
    corners: Vec<f32>,
    /// Block-local index of the vertex on each edge, [`NO_VERTEX`] if none;
    /// see [`EDGE_SLOT_OFFSETS`].
    vertex_slots: Vec<u32>,
    /// The slots set by the current block, reset after it.
    used_slots: Vec<usize>,
}

impl MeshScratch {
    fn new() -> Self {
        Self {
            corners: vec![f32::NAN; SPAN * SPAN * SPAN],
            vertex_slots: vec![NO_VERTEX; SPAN * SPAN * SPAN * 3],
            used_slots: Vec::with_capacity(512),
        }
    }
}

/// Triangles of the cells anchored in block `key`. A cell yields triangles
/// only when all eight corners are observed (nonzero weight, at least
/// `min_weight`, finite value).
fn block_mesh(
    key: BlockKey,
    blocks: &Blocks,
    min_weight: f32,
    scratch: &mut MeshScratch,
) -> BlockMesh {
    let neighbors = neighbor_blocks(key, blocks);
    if !has_possible_surface(&neighbors) {
        return BlockMesh::default();
    }
    let Some(origin) = checked_block_origin(key) else {
        return BlockMesh::default();
    };
    gather_corners(&neighbors, min_weight, &mut scratch.corners);
    let MeshScratch {
        corners,
        vertex_slots,
        used_slots,
    } = scratch;
    let mut mesh = BlockMesh {
        vertices: Vec::with_capacity(256),
        triangles: Vec::with_capacity(512),
    };
    for z in 0..BLOCK_RESOLUTION_U16 {
        for y in 0..BLOCK_RESOLUTION_U16 {
            for x in 0..BLOCK_RESOLUTION_U16 {
                let cell = (usize::from(z) * SPAN + usize::from(y)) * SPAN + usize::from(x);
                let values = CORNER_OFFSETS.map(|offset| corners[cell + offset]);
                if values.iter().any(|value| value.is_nan()) {
                    continue;
                }
                let cube_index = values
                    .iter()
                    .enumerate()
                    .fold(0_u8, |index, (corner, value)| {
                        index | (u8::from(*value < 0.0) << corner)
                    });
                if cube_index == 0 || cube_index == u8::MAX {
                    continue;
                }
                let anchor = add(origin, [x, y, z].map(i64::from));
                for_each_triangle(cube_index, |edges| {
                    let triangle = edges.map(|edge_index| {
                        let slot = cell * 3 + EDGE_SLOT_OFFSETS[edge_index];
                        if vertex_slots[slot] == NO_VERTEX {
                            let edge = &EDGES[edge_index];
                            let key = EdgeKey {
                                base: add(anchor, edge.offset),
                                axis: edge.axis,
                            };
                            let [start, end] = edge.corners;
                            // A block has fewer than 3 * 17^3 edges.
                            vertex_slots[slot] = mesh.vertices.len() as u32;
                            used_slots.push(slot);
                            mesh.vertices.push(EdgeVertex {
                                edge: key,
                                amount: interpolation_amount(values[start], values[end]),
                                on_seam: on_block_seam(key, origin),
                            });
                        }
                        vertex_slots[slot]
                    });
                    mesh.triangles.push(triangle);
                });
            }
        }
    }
    for slot in used_slots.drain(..) {
        vertex_slots[slot] = NO_VERTEX;
    }
    mesh
}

/// Fill `corners` ([`SPAN`]-cubed, x fastest) with the TSDF of each voxel
/// the block's cells read, NaN where it is unobserved, fused fewer than
/// `min_weight` times, or not finite.
fn gather_corners(neighbors: &[Option<&VoxelBlock>; 8], min_weight: f32, corners: &mut [f32]) {
    let split = |coordinate: usize| {
        (
            coordinate / BLOCK_RESOLUTION_USIZE,
            coordinate % BLOCK_RESOLUTION_USIZE,
        )
    };
    for z in 0..SPAN {
        let (block_z, local_z) = split(z);
        for y in 0..SPAN {
            let (block_y, local_y) = split(y);
            let row = (z * SPAN + y) * SPAN;
            for x in 0..SPAN {
                let (block_x, local_x) = split(x);
                let value = neighbors[block_x | block_y << 1 | block_z << 2].and_then(|block| {
                    let index = voxel_index([local_x, local_y, local_z]);
                    let (value, weight) = (block.tsdf[index], block.weights[index]);
                    (weight != 0.0 && weight >= min_weight && value.is_finite()).then_some(value)
                });
                corners[row + x] = value.unwrap_or(f32::NAN);
            }
        }
    }
}

/// Fraction of the way from sample `a` to sample `b` where the linear
/// interpolant crosses zero.
pub(crate) fn interpolation_amount(value_a: f32, value_b: f32) -> f32 {
    if value_a == 0.0 {
        0.0
    } else if value_b == 0.0 {
        1.0
    } else {
        value_a / (value_a - value_b)
    }
}

/// Whether cells outside the block at `origin` also contain `edge`: cells
/// anchored at `base - {0,1}` across the edge axis and at `base` along it.
fn on_block_seam(edge: EdgeKey, origin: [i64; 3]) -> bool {
    (0..3).any(|axis| {
        let local = edge.base[axis] - origin[axis];
        let first = if axis == usize::from(edge.axis) { 0 } else { 1 };
        !(first..BLOCK_RESOLUTION).contains(&local)
    })
}

/// The 2x2x2 blocks a block's cells read: `key + (i & 1, i >> 1 & 1, i >> 2 & 1)`.
fn neighbor_blocks(key: BlockKey, blocks: &Blocks) -> [Option<&VoxelBlock>; 8] {
    std::array::from_fn(|index| {
        let offset = [index & 1, (index >> 1) & 1, (index >> 2) & 1].map(|bit| bit as i64);
        blocks.get(&checked_offset(key, offset)?)
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

/// The block holding global voxel `coordinate` and the voxel's index in it.
fn voxel_location(coordinate: [i64; 3]) -> (BlockKey, usize) {
    let key = coordinate.map(|value| value.div_euclid(BLOCK_RESOLUTION));
    let local = coordinate.map(|value| value.rem_euclid(BLOCK_RESOLUTION));
    (key, local_index(local))
}

/// Voxel lookup by global coordinate.
trait Voxels {
    /// The allocated block holding global voxel `coordinate`, and the
    /// voxel's index in it.
    fn voxel(&self, coordinate: [i64; 3]) -> Option<(&VoxelBlock, usize)>;
}

impl Voxels for Blocks {
    fn voxel(&self, coordinate: [i64; 3]) -> Option<(&VoxelBlock, usize)> {
        let (key, index) = voxel_location(coordinate);
        self.get(&key).map(|block| (block, index))
    }
}

/// A block and its 26 neighbors, looked up once: what its vertices' colors
/// and normals read. Other coordinates fall back to the map.
struct Neighborhood<'a> {
    /// Global coordinate of the first voxel of the block at offset `(-1, -1, -1)`.
    first: [i64; 3],
    /// The blocks at `NEIGHBORHOOD` offsets, x fastest.
    blocks: [Option<&'a VoxelBlock>; 27],
    all: &'a Blocks,
}

impl<'a> Neighborhood<'a> {
    fn new(key: BlockKey, all: &'a Blocks) -> Option<Self> {
        let origin = checked_block_origin(key)?;
        let first = checked_offset(origin, [-BLOCK_RESOLUTION; 3])?;
        let blocks =
            std::array::from_fn(|index| all.get(&checked_offset(key, NEIGHBORHOOD[index])?));
        Some(Self { first, blocks, all })
    }
}

impl Voxels for Neighborhood<'_> {
    fn voxel(&self, coordinate: [i64; 3]) -> Option<(&VoxelBlock, usize)> {
        let span = 3 * BLOCK_RESOLUTION;
        let relative = [
            coordinate[0].wrapping_sub(self.first[0]),
            coordinate[1].wrapping_sub(self.first[1]),
            coordinate[2].wrapping_sub(self.first[2]),
        ];
        if !relative.iter().all(|value| (0..span).contains(value)) {
            return self.all.voxel(coordinate);
        }
        let [x, y, z] = relative.map(|value| value / BLOCK_RESOLUTION);
        let block = self.blocks[(x + 3 * y + 9 * z) as usize]?;
        Some((
            block,
            local_index(relative.map(|value| value % BLOCK_RESOLUTION)),
        ))
    }
}

/// TSDF of global voxel `coordinate`, when observed (nonzero weight, finite).
fn observed_tsdf(voxels: &impl Voxels, coordinate: [i64; 3]) -> Option<f32> {
    let (block, index) = voxels.voxel(coordinate)?;
    let value = block.tsdf[index];
    (block.weights[index] != 0.0 && value.is_finite()).then_some(value)
}

/// Unit TSDF gradient at a vertex: central differences at both edge
/// endpoints, linearly interpolated along the edge. The gradient points from
/// inside (negative) to outside, toward the cameras. Reading neighbors across
/// block borders makes the normal a function of the edge alone, so per-block
/// meshes agree with the full mesh at seams.
fn vertex_normal(vertex: &EdgeVertex, voxels: &impl Voxels) -> [f32; 3] {
    let start = gradient(voxels, vertex.edge.base);
    let end = gradient(voxels, vertex.edge.end());
    // A vertex on a voxel (amount 0 or 1) takes that voxel's gradient
    // exactly, so the edges that meet there agree on its normal.
    let normal = match vertex.amount {
        0.0 => start,
        1.0 => end,
        t => std::array::from_fn(|axis| start[axis] + (end[axis] - start[axis]) * t),
    };
    let magnitude = length(normal);
    if magnitude > 0.0 && magnitude.is_finite() {
        return normal.map(|component| component / magnitude);
    }
    // Degenerate gradient: fall back to the edge direction. The edge crosses
    // zero, so its endpoints differ and give the outward sign.
    let mut fallback = [0.0; 3];
    let rising = observed_tsdf(voxels, vertex.edge.end()) > observed_tsdf(voxels, vertex.edge.base);
    fallback[usize::from(vertex.edge.axis)] = if rising { 1.0 } else { -1.0 };
    fallback
}

/// Central-difference TSDF gradient at a voxel (half the difference of its
/// two neighbors per axis), one-sided where a neighbor is unobserved, zero
/// where both are.
fn gradient(voxels: &impl Voxels, coordinate: [i64; 3]) -> [f32; 3] {
    let center = observed_tsdf(voxels, coordinate);
    std::array::from_fn(|axis| {
        let mut plus = coordinate;
        plus[axis] += 1;
        let mut minus = coordinate;
        minus[axis] -= 1;
        match (
            observed_tsdf(voxels, minus),
            observed_tsdf(voxels, plus),
            center,
        ) {
            (Some(minus), Some(plus), _) => (plus - minus) * 0.5,
            (None, Some(plus), Some(center)) => plus - center,
            (Some(minus), None, Some(center)) => center - minus,
            _ => 0.0,
        }
    })
}

/// RGB of the voxel nearest `position`; black when that voxel is not allocated.
fn nearest_color(position: [f32; 3], voxels: &impl Voxels, voxel_size: f32) -> [u8; 3] {
    let voxel_size = f64::from(voxel_size);
    let coordinate = position.map(|value| (f64::from(value) / voxel_size - 0.5).round());
    let Some(coordinate) = checked_coordinate(coordinate) else {
        return [0; 3];
    };
    voxels
        .voxel(coordinate)
        .and_then(|(block, index)| Some((block.color.as_ref()?, index)))
        .map_or([0; 3], |(color, index)| {
            color.rgb[index].map(|value| value.round().clamp(0.0, 255.0) as u8)
        })
}

fn checked_coordinate(values: [f64; 3]) -> Option<[i64; 3]> {
    Some([
        crate::geometry::checked_trunc_i64(values[0])?,
        crate::geometry::checked_trunc_i64(values[1])?,
        crate::geometry::checked_trunc_i64(values[2])?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(blocks: &Blocks) -> VolumeView<'_> {
        VolumeView {
            blocks,
            voxel_size: 0.04,
            fuse_color: false,
            min_weight: 0.0,
        }
    }

    #[test]
    fn a_cell_with_a_nan_corner_is_unobserved() {
        let mut block = VoxelBlock::new(false);
        for offset in CORNERS {
            block.weights[local_index(offset)] = 1.0;
        }
        block.tsdf[local_index(CORNERS[0])] = -1.0;
        block.tsdf[local_index(CORNERS[1])] = f32::NAN;
        block.observed_min = -1.0;
        block.observed_max = 1.0;
        let mut blocks = Blocks::default();
        blocks.insert([0, 0, 0], block);

        let mesh = full_mesh(&view(&blocks)).expect("small mesh");

        assert!(mesh.vertices.is_empty());
        assert!(mesh.triangles.is_empty());
    }

    #[test]
    fn the_lowest_block_key_meshes_without_overflow() {
        let mut block = VoxelBlock::new(false);
        block.weights.fill(1.0);
        block.tsdf[local_index([0, 0, 0])] = -1.0;
        block.observed_min = -1.0;
        block.observed_max = 1.0;
        let mut blocks = Blocks::default();
        blocks.insert([MIN_BLOCK_INDEX; 3], block);

        // The normals read one voxel below the block's first voxel.
        let mesh = full_mesh(&view(&blocks)).expect("small mesh");

        assert_eq!(mesh.triangles.len(), 1);
    }

    #[test]
    fn seam_edges_are_exactly_those_shared_with_other_blocks() {
        let origin = [16, -32, 0];
        // An x edge at local (15, 1, 1) belongs to cells (15, 0..=1, 0..=1): interior.
        let interior = EdgeKey {
            base: [origin[0] + 15, origin[1] + 1, origin[2] + 1],
            axis: 0,
        };
        assert!(!on_block_seam(interior, origin));
        // A y edge at local x = 0 is shared with the cell at x = -1 in the previous block.
        let lower = EdgeKey {
            base: [origin[0], origin[1] + 3, origin[2] + 3],
            axis: 1,
        };
        assert!(on_block_seam(lower, origin));
        // A z edge at local x = 16 belongs to the next block's cells too.
        let upper = EdgeKey {
            base: [origin[0] + 16, origin[1] + 3, origin[2] + 3],
            axis: 2,
        };
        assert!(on_block_seam(upper, origin));
    }
}
