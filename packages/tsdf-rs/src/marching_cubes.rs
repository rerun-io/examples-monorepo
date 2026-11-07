use std::collections::hash_map::Entry;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::error::TsdfError;
use crate::tables::{EDGE_TABLE, TRI_TABLE};

const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

const EDGE_GEOMETRY: [[usize; 4]; 12] = [
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

#[derive(Clone, Copy)]
enum GridLayout {
    Xyz,
}

/// Triangle mesh extracted from a zero level set.
///
/// Vertex positions use world-space units. Triangle rows contain zero-based
/// indices into `vertices`. `colors` is present only for a color-fusing volume.
///
/// # Example
///
/// ```
/// use tsdf_rs::marching_cubes;
///
/// let values = vec![1.0_f32; 8];
/// let mesh = marching_cubes(&values, [2, 2, 2], 0.04, [0.0; 3])?;
/// assert!(mesh.vertices.is_empty());
/// # Ok::<(), tsdf_rs::TsdfError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    /// World-space vertex positions in the same units as the voxel size.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle vertex indices with counter-clockwise table winding.
    pub triangles: Vec<[u32; 3]>,
    /// Per-vertex RGB colors, when color fusion is enabled.
    pub colors: Option<Vec<[u8; 3]>>,
}

impl Mesh {
    fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
            colors: None,
        }
    }
}

/// Extract a triangle mesh from a dense XYZ-ordered TSDF grid.
///
/// The input is C-contiguous with the Z coordinate varying fastest. A grid
/// sample at integer coordinate `(x, y, z)` lies at
/// `origin + voxel_size * (x, y, z)`. Linear interpolation places vertices on
/// zero crossings.
///
/// # Arguments
///
/// * `values` - TSDF samples with shape `dimensions` and Z varying fastest.
/// * `dimensions` - Grid dimensions `[nx, ny, nz]`.
/// * `voxel_size` - Distance between neighboring grid samples, in world units.
/// * `origin` - World-space position of sample `(0, 0, 0)`.
///
/// # Returns
///
/// A deterministic mesh of the zero level set.
///
/// # Errors
///
/// Returns [`TsdfError::InvalidParameter`] for a non-positive or non-finite
/// voxel size. Returns [`TsdfError::ShapeMismatch`] when `values` does not
/// match `dimensions`, or [`TsdfError::DimensionOverflow`] when the dimension
/// product overflows.
///
/// # Example
///
/// ```
/// use tsdf_rs::marching_cubes;
///
/// let values = vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
/// let mesh = marching_cubes(&values, [2, 2, 2], 0.5, [1.0, 2.0, 3.0])?;
/// assert!(!mesh.vertices.is_empty());
/// # Ok::<(), tsdf_rs::TsdfError>(())
/// ```
pub fn marching_cubes(
    values: &[f32],
    dimensions: [usize; 3],
    voxel_size: f32,
    origin: [f32; 3],
) -> Result<Mesh, TsdfError> {
    validate_voxel_size(voxel_size)?;
    let expected = checked_element_count("values", &dimensions)?;
    if values.len() != expected {
        return Err(TsdfError::ShapeMismatch {
            name: "values",
            expected: dimensions.to_vec(),
            got: vec![values.len()],
        });
    }
    extract(
        values,
        dimensions,
        GridLayout::Xyz,
        voxel_size,
        origin,
        None,
    )
}

fn extract(
    values: &[f32],
    dimensions: [usize; 3],
    layout: GridLayout,
    voxel_size: f32,
    origin: [f32; 3],
    weights: Option<&[u16]>,
) -> Result<Mesh, TsdfError> {
    let [nx, ny, nz] = dimensions;
    if nx < 2 || ny < 2 || nz < 2 {
        return Ok(Mesh::empty());
    }

    let slab_triangles: Result<Vec<Vec<[u64; 3]>>, TsdfError> = (0..nz - 1)
        .into_par_iter()
        .map(|z| {
            let mut triangles = Vec::new();
            for x in 0..nx - 1 {
                for y in 0..ny - 1 {
                    let mut cube_index = 0usize;
                    let mut observed = true;
                    for (corner, offset) in CORNERS.iter().enumerate() {
                        let index = grid_index(
                            x + offset[0],
                            y + offset[1],
                            z + offset[2],
                            dimensions,
                            layout,
                        );
                        if weights.is_some_and(|all_weights| all_weights[index] == 0) {
                            observed = false;
                            break;
                        }
                        let value = values[index];
                        if !value.is_finite() {
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
                        // Emit vertices 0,2,1: the classic tables wind clockwise
                        // seen from outside under this corner/sign convention,
                        // and Open3D's counter-clockwise winding is the contract.
                        triangles.push([
                            edge_id(x, y, z, triangle_edges[0] as usize, dimensions)?,
                            edge_id(x, y, z, triangle_edges[2] as usize, dimensions)?,
                            edge_id(x, y, z, triangle_edges[1] as usize, dimensions)?,
                        ]);
                    }
                }
            }
            Ok(triangles)
        })
        .collect();

    merge_dense_edges(
        slab_triangles?.into_iter().flatten(),
        values,
        dimensions,
        layout,
        voxel_size,
        origin,
    )
}

fn merge_dense_edges(
    edge_triangles: impl Iterator<Item = [u64; 3]>,
    values: &[f32],
    dimensions: [usize; 3],
    layout: GridLayout,
    voxel_size: f32,
    origin: [f32; 3],
) -> Result<Mesh, TsdfError> {
    let mut edge_to_vertex = FxHashMap::<u64, u32>::default();
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

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
                    vertices.push(interpolate_dense_edge(
                        edge, values, dimensions, layout, voxel_size, origin,
                    ));
                    entry.insert(vertex_index);
                    vertex_index
                }
            };
        }
        triangles.push(triangle);
    }

    Ok(Mesh {
        vertices,
        triangles,
        colors: None,
    })
}

fn grid_index(x: usize, y: usize, z: usize, dimensions: [usize; 3], layout: GridLayout) -> usize {
    let [_nx, ny, nz] = dimensions;
    match layout {
        GridLayout::Xyz => (x * ny + y) * nz + z,
    }
}

fn edge_id(
    x: usize,
    y: usize,
    z: usize,
    edge: usize,
    dimensions: [usize; 3],
) -> Result<u64, TsdfError> {
    let geometry = EDGE_GEOMETRY
        .get(edge)
        .ok_or(TsdfError::InvalidEdge { edge })?;
    let base_x = x + geometry[0];
    let base_y = y + geometry[1];
    let base_z = z + geometry[2];
    let point_id = (base_x * dimensions[1] + base_y) * dimensions[2] + base_z;
    Ok((point_id * 3 + geometry[3]) as u64)
}

fn decode_edge(edge: u64, dimensions: [usize; 3]) -> ([usize; 3], usize) {
    let edge = edge as usize;
    let axis = edge % 3;
    let point_id = edge / 3;
    let z = point_id % dimensions[2];
    let xy = point_id / dimensions[2];
    let y = xy % dimensions[1];
    let x = xy / dimensions[1];
    ([x, y, z], axis)
}

fn interpolate_dense_edge(
    edge: u64,
    values: &[f32],
    dimensions: [usize; 3],
    layout: GridLayout,
    voxel_size: f32,
    origin: [f32; 3],
) -> [f32; 3] {
    let (base, axis) = decode_edge(edge, dimensions);
    let mut other = base;
    other[axis] += 1;
    let value_a = values[grid_index(base[0], base[1], base[2], dimensions, layout)];
    let value_b = values[grid_index(other[0], other[1], other[2], dimensions, layout)];
    let amount = interpolation_amount(value_a, value_b);

    let mut grid_position = [base[0] as f32, base[1] as f32, base[2] as f32];
    grid_position[axis] += amount;
    [
        origin[0] + grid_position[0] * voxel_size,
        origin[1] + grid_position[1] * voxel_size,
        origin[2] + grid_position[2] * voxel_size,
    ]
}

pub(crate) fn interpolation_amount(value_a: f32, value_b: f32) -> f32 {
    if value_a == 0.0 {
        0.0
    } else if value_b == 0.0 {
        1.0
    } else {
        value_a / (value_a - value_b)
    }
}

fn validate_voxel_size(voxel_size: f32) -> Result<(), TsdfError> {
    if voxel_size.is_finite() && voxel_size > 0.0 {
        Ok(())
    } else {
        Err(TsdfError::InvalidParameter {
            name: "voxel_size",
            reason: "must be finite and positive",
            value: voxel_size,
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
