//! Marching-cubes extraction over the sparse blocks: the full mesh and per-block meshes.

use std::collections::hash_map::Entry;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::TsdfError;
use crate::mesh::{interpolation_amount, Mesh};
use crate::tables::{for_each_triangle, CORNERS, EDGES};
use crate::volume::{checked_block_origin, local_index, BlockKey, VoxelBlock, BLOCK_RESOLUTION};

type Blocks = FxHashMap<BlockKey, VoxelBlock>;

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

/// Full extraction: blocks mesh in parallel, then a sequential merge in
/// sorted block order joins vertices on block seams. The vertex order is the
/// first-seen order of the whole sorted triangle stream.
pub(crate) fn full_mesh(
    blocks: &Blocks,
    voxel_size: f32,
    fuse_color: bool,
) -> Result<Mesh, TsdfError> {
    let mut keys: Vec<BlockKey> = blocks.keys().copied().collect();
    keys.sort_unstable();
    let block_meshes: Vec<BlockMesh> = keys
        .par_iter()
        .map(|&key| block_mesh(key, blocks))
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
    Ok(attributed_mesh(
        &vertices, triangles, blocks, voxel_size, fuse_color,
    ))
}

/// Meshes of `keys` (sorted), keeping empty ones only for `surfaced` blocks.
pub(crate) fn block_meshes(
    keys: &[BlockKey],
    blocks: &Blocks,
    surfaced: &FxHashSet<BlockKey>,
    voxel_size: f32,
    fuse_color: bool,
) -> Vec<(BlockKey, Mesh)> {
    keys.par_iter()
        .filter_map(|&key| {
            let block = block_mesh(key, blocks);
            // Report empty meshes only on the surfaced -> empty transition, so
            // surface-less shell blocks are not re-announced every extraction.
            if block.triangles.is_empty() && !surfaced.contains(&key) {
                return None;
            }
            let mesh = attributed_mesh(
                &block.vertices,
                block.triangles,
                blocks,
                voxel_size,
                fuse_color,
            );
            Some((key, mesh))
        })
        .collect()
}

/// Positions, colors and normals of `vertices`, computed in parallel.
fn attributed_mesh(
    vertices: &[EdgeVertex],
    triangles: Vec<[u32; 3]>,
    blocks: &Blocks,
    voxel_size: f32,
    fuse_color: bool,
) -> Mesh {
    let positions: Vec<[f32; 3]> = vertices
        .par_iter()
        .map(|vertex| {
            let mut position = vertex
                .edge
                .base
                .map(|coordinate| (coordinate as f32 + 0.5) * voxel_size);
            position[usize::from(vertex.edge.axis)] += vertex.amount * voxel_size;
            position
        })
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
    Mesh {
        vertices: positions,
        triangles,
        colors,
        normals,
    }
}

/// Triangles of the cells anchored in block `key`. A cell yields triangles
/// only when all eight corners are observed (nonzero weight, finite value).
fn block_mesh(key: BlockKey, blocks: &Blocks) -> BlockMesh {
    let neighbors = neighbor_blocks(key, blocks);
    if !has_possible_surface(&neighbors) {
        return BlockMesh::default();
    }
    let Some(origin) = checked_block_origin(key) else {
        return BlockMesh::default();
    };
    let mut mesh = BlockMesh {
        vertices: Vec::with_capacity(256),
        triangles: Vec::with_capacity(512),
    };
    let mut local_vertices = FxHashMap::<EdgeKey, u32>::default();
    for z in 0..BLOCK_RESOLUTION {
        for y in 0..BLOCK_RESOLUTION {
            for x in 0..BLOCK_RESOLUTION {
                let Some((cube_index, values)) = observed_cell(&neighbors, [x, y, z]) else {
                    continue;
                };
                if cube_index == 0 || cube_index == u8::MAX {
                    continue;
                }
                let anchor = [origin[0] + x, origin[1] + y, origin[2] + z];
                for_each_triangle(cube_index, |edges| {
                    let triangle = edges.map(|edge_index| {
                        let edge = &EDGES[edge_index];
                        let key = EdgeKey {
                            base: [
                                anchor[0] + edge.offset[0],
                                anchor[1] + edge.offset[1],
                                anchor[2] + edge.offset[2],
                            ],
                            axis: edge.axis,
                        };
                        *local_vertices.entry(key).or_insert_with(|| {
                            let [start, end] = edge.corners;
                            mesh.vertices.push(EdgeVertex {
                                edge: key,
                                amount: interpolation_amount(values[start], values[end]),
                                on_seam: on_block_seam(key, origin),
                            });
                            // A block has fewer than 3 * 17^3 edges.
                            (mesh.vertices.len() - 1) as u32
                        })
                    });
                    mesh.triangles.push(triangle);
                });
            }
        }
    }
    mesh
}

/// Configuration and corner values of the cell anchored at block-local
/// `anchor`; `None` if a corner is unobserved.
fn observed_cell(neighbors: &[Option<&VoxelBlock>; 8], anchor: [i64; 3]) -> Option<(u8, [f32; 8])> {
    let mut cube_index = 0_u8;
    let mut values = [0.0_f32; 8];
    for (corner, offset) in CORNERS.iter().enumerate() {
        let local = [
            anchor[0] + offset[0],
            anchor[1] + offset[1],
            anchor[2] + offset[2],
        ];
        let (value, weight) = cached_voxel(neighbors, local)?;
        if weight == 0.0 || !value.is_finite() {
            return None;
        }
        values[corner] = value;
        if value < 0.0 {
            cube_index |= 1 << corner;
        }
    }
    Some((cube_index, values))
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

/// `(tsdf, weight)` of block-local voxel `local` (each coordinate in `0..32`).
fn cached_voxel(neighbors: &[Option<&VoxelBlock>; 8], local: [i64; 3]) -> Option<(f32, f32)> {
    let neighbor_index = (local[0] / BLOCK_RESOLUTION)
        | ((local[1] / BLOCK_RESOLUTION) << 1)
        | ((local[2] / BLOCK_RESOLUTION) << 2);
    let block = neighbors[neighbor_index as usize]?;
    let index = local_index(local.map(|coordinate| coordinate % BLOCK_RESOLUTION));
    Some((block.tsdf[index], block.weights[index]))
}

/// The block holding global voxel `coordinate` and the voxel's index in it.
fn voxel_location(coordinate: [i64; 3]) -> (BlockKey, usize) {
    let key = coordinate.map(|value| value.div_euclid(BLOCK_RESOLUTION));
    let local = coordinate.map(|value| value.rem_euclid(BLOCK_RESOLUTION));
    (key, local_index(local))
}

/// TSDF of global voxel `coordinate`, when observed (nonzero weight, finite).
fn observed_tsdf(blocks: &Blocks, coordinate: [i64; 3]) -> Option<f32> {
    let (key, index) = voxel_location(coordinate);
    let block = blocks.get(&key)?;
    let value = block.tsdf[index];
    (block.weights[index] != 0.0 && value.is_finite()).then_some(value)
}

/// Unit TSDF gradient at a vertex: central differences at both edge
/// endpoints, linearly interpolated along the edge. The gradient points from
/// inside (negative) to outside, toward the cameras. Reading neighbors across
/// block borders makes the normal a function of the edge alone, so per-block
/// meshes agree with the full mesh at seams.
fn vertex_normal(vertex: &EdgeVertex, blocks: &Blocks) -> [f32; 3] {
    let start = gradient(blocks, vertex.edge.base);
    let end = gradient(blocks, vertex.edge.end());
    // A vertex on a voxel (amount 0 or 1) takes that voxel's gradient
    // exactly, so the edges that meet there agree on its normal.
    let normal = match vertex.amount {
        0.0 => start,
        1.0 => end,
        t => std::array::from_fn(|axis| start[axis] + (end[axis] - start[axis]) * t),
    };
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if length > 0.0 && length.is_finite() {
        return normal.map(|component| component / length);
    }
    // Degenerate gradient: fall back to the edge direction. The edge crosses
    // zero, so its endpoints differ and give the outward sign.
    let mut fallback = [0.0; 3];
    let rising = observed_tsdf(blocks, vertex.edge.end()) > observed_tsdf(blocks, vertex.edge.base);
    fallback[usize::from(vertex.edge.axis)] = if rising { 1.0 } else { -1.0 };
    fallback
}

/// Central-difference TSDF gradient at a voxel (half the difference of its
/// two neighbors per axis), one-sided where a neighbor is unobserved, zero
/// where both are.
fn gradient(blocks: &Blocks, coordinate: [i64; 3]) -> [f32; 3] {
    let center = observed_tsdf(blocks, coordinate);
    std::array::from_fn(|axis| {
        let mut plus = coordinate;
        plus[axis] += 1;
        let mut minus = coordinate;
        minus[axis] -= 1;
        match (
            observed_tsdf(blocks, minus),
            observed_tsdf(blocks, plus),
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
fn nearest_color(position: [f32; 3], blocks: &Blocks, voxel_size: f32) -> [u8; 3] {
    let voxel_size = f64::from(voxel_size);
    let coordinate = position.map(|value| (f64::from(value) / voxel_size - 0.5).round());
    let Some(coordinate) = checked_coordinate(coordinate) else {
        return [0; 3];
    };
    let (key, index) = voxel_location(coordinate);
    blocks
        .get(&key)
        .and_then(|block| block.color.as_ref())
        .map_or([0; 3], |color| {
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

        let mesh = full_mesh(&blocks, 0.04, false).expect("small mesh");

        assert!(mesh.vertices.is_empty());
        assert!(mesh.triangles.is_empty());
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
