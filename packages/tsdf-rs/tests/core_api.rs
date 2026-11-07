use std::collections::HashMap;
use std::error::Error;

use tsdf_rs::{marching_cubes, Mesh, TsdfError, TsdfVolume};

#[test]
fn integration_reports_depth_shape_mismatch() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let result = volume.integrate(
        &[1.0; 3],
        [2, 2],
        [100.0, 0.0, 0.5, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0],
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        None,
    );

    match result {
        Err(TsdfError::ShapeMismatch {
            name,
            expected,
            got,
        }) => {
            assert_eq!(name, "depth");
            assert_eq!(expected, vec![2, 2]);
            assert_eq!(got, vec![3]);
            Ok(())
        }
        other => Err(format!("expected ShapeMismatch, got {other:?}").into()),
    }
}

#[test]
fn mesh_bytes_are_invariant_to_rayon_thread_count() -> Result<(), Box<dyn Error>> {
    let mesh_one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()?
        .install(build_plane_mesh)?;
    let mesh_eight = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()?
        .install(build_plane_mesh)?;

    assert_eq!(mesh_one, mesh_eight);
    assert!(!mesh_one.vertices.is_empty());
    Ok(())
}

fn build_plane_mesh() -> Result<tsdf_rs::Mesh, TsdfError> {
    let height = 96;
    let width = 128;
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let depth = vec![2.0_f32; height * width];
    let intrinsics = [
        100.0,
        0.0,
        width as f32 / 2.0 - 0.5,
        0.0,
        100.0,
        height as f32 / 2.0 - 0.5,
        0.0,
        0.0,
        1.0,
    ];
    let transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    volume.integrate(&depth, [height, width], intrinsics, transform, None)?;
    volume.extract_mesh()
}

#[test]
fn repeated_observations_use_a_cumulative_average() -> Result<(), Box<dyn Error>> {
    const INITIAL_OBSERVATIONS: usize = 1_000;
    const SHIFTED_OBSERVATIONS: usize = 100;
    const INITIAL_DEPTH: f32 = 1.0;
    const SHIFTED_DEPTH: f32 = 1.1;

    let height = 8;
    let width = 8;
    let mut volume = TsdfVolume::new(0.01, 0.12, false)?;
    let intrinsics = [
        100.0,
        0.0,
        width as f32 / 2.0 - 0.5,
        0.0,
        100.0,
        height as f32 / 2.0 - 0.5,
        0.0,
        0.0,
        1.0,
    ];
    let transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let initial_depth = vec![INITIAL_DEPTH; height * width];
    let shifted_depth = vec![SHIFTED_DEPTH; height * width];

    for _ in 0..INITIAL_OBSERVATIONS {
        volume.integrate(&initial_depth, [height, width], intrinsics, transform, None)?;
    }
    for _ in 0..SHIFTED_OBSERVATIONS {
        volume.integrate(&shifted_depth, [height, width], intrinsics, transform, None)?;
    }

    let mesh = volume.extract_mesh()?;
    assert!(!mesh.vertices.is_empty());
    let expected_depth = (INITIAL_DEPTH * INITIAL_OBSERVATIONS as f32
        + SHIFTED_DEPTH * SHIFTED_OBSERVATIONS as f32)
        / (INITIAL_OBSERVATIONS + SHIFTED_OBSERVATIONS) as f32;
    let maximum_error = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex[2] - expected_depth).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum_error < 0.01,
        "surface differs from cumulative-average depth {expected_depth} by {maximum_error}"
    );
    Ok(())
}

#[test]
fn dense_cube_with_a_nan_sample_is_unobserved() -> Result<(), TsdfError> {
    let values = [-1.0, 1.0, 1.0, 1.0, f32::NAN, 1.0, 1.0, 1.0];

    let mesh = marching_cubes(&values, [2, 2, 2], 0.04, [0.0; 3])?;

    assert!(mesh.vertices.is_empty());
    assert!(mesh.triangles.is_empty());
    Ok(())
}

/// Bit-exact triangle multiset of a mesh: each triangle as its three vertex
/// positions' f32 bit patterns, sorted so vertex-index layout differences
/// (global dedup vs per-block dedup) cannot affect the comparison.
fn triangle_multiset(mesh: &Mesh) -> Vec<[[u32; 3]; 3]> {
    let mut triangles: Vec<[[u32; 3]; 3]> = mesh
        .triangles
        .iter()
        .map(|triangle| triangle.map(|index| mesh.vertices[index as usize].map(f32::to_bits)))
        .collect();
    triangles.sort_unstable();
    triangles
}

fn centered_intrinsics(height: usize, width: usize) -> [f32; 9] {
    [
        100.0,
        0.0,
        width as f32 / 2.0 - 0.5,
        0.0,
        100.0,
        height as f32 / 2.0 - 0.5,
        0.0,
        0.0,
        1.0,
    ]
}

fn wavy_frame(height: usize, width: usize, frame: usize) -> (Vec<f32>, [f32; 16]) {
    let depth = (0..height * width)
        .map(|index| {
            let u = (index % width) as f32;
            2.0 + 0.03 * frame as f32 + 0.1 * (u / 9.0).sin()
        })
        .collect();
    // Translate the camera along x each frame so new blocks appear beside
    // blocks that are simultaneously being updated.
    let transform = [
        1.0,
        0.0,
        0.0,
        -0.4 * frame as f32,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    (depth, transform)
}

#[test]
fn dirty_block_union_matches_full_extraction_after_every_frame() -> Result<(), Box<dyn Error>> {
    let height = 96;
    let width = 128;
    let intrinsics = centered_intrinsics(height, width);
    let mut volume = TsdfVolume::new(0.04, 0.12, true)?;
    let mut latest: HashMap<[i64; 3], Mesh> = HashMap::new();
    let color: Vec<u8> = (0..height * width * 3)
        .map(|index| (index % 251) as u8)
        .collect();

    for frame in 0..3 {
        let (depth, transform) = wavy_frame(height, width, frame);
        volume.integrate(&depth, [height, width], intrinsics, transform, Some(&color))?;
        for (key, mesh) in volume.extract_dirty_block_meshes()? {
            latest.insert(key, mesh);
        }

        let full = volume.extract_mesh()?;
        let mut union_triangles: Vec<[[u32; 3]; 3]> =
            latest.values().flat_map(triangle_multiset).collect();
        union_triangles.sort_unstable();
        assert_eq!(
            union_triangles,
            triangle_multiset(&full),
            "per-block union diverged from full extraction at frame {frame}"
        );

        // Colors are position-determined, so block meshes must agree with the
        // full mesh wherever positions coincide.
        let full_colors: HashMap<[u32; 3], [u8; 3]> = full
            .vertices
            .iter()
            .zip(full.colors.as_ref().ok_or("full mesh missing colors")?)
            .map(|(vertex, color)| (vertex.map(f32::to_bits), *color))
            .collect();
        for mesh in latest.values() {
            let colors = mesh.colors.as_ref().ok_or("block mesh missing colors")?;
            for (vertex, color) in mesh.vertices.iter().zip(colors) {
                assert_eq!(
                    full_colors.get(&vertex.map(f32::to_bits)),
                    Some(color),
                    "block vertex color diverged from full mesh at frame {frame}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn extract_dirty_block_meshes_drains_the_dirty_set() -> Result<(), Box<dyn Error>> {
    let height = 96;
    let width = 128;
    let intrinsics = centered_intrinsics(height, width);
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let (depth, transform) = wavy_frame(height, width, 0);

    volume.integrate(&depth, [height, width], intrinsics, transform, None)?;
    let first = volume.extract_dirty_block_meshes()?;
    assert!(!first.is_empty());
    // Blocks that never held surface are omitted, not reported as empty.
    assert!(first.iter().all(|(_, mesh)| !mesh.triangles.is_empty()));
    assert!(volume.extract_dirty_block_meshes()?.is_empty());

    volume.integrate(&depth, [height, width], intrinsics, transform, None)?;
    assert!(!volume.extract_dirty_block_meshes()?.is_empty());
    Ok(())
}

#[test]
fn dirty_block_meshes_are_invariant_to_rayon_thread_count() -> Result<(), Box<dyn Error>> {
    let run = || -> Result<Vec<([i64; 3], Mesh)>, TsdfError> {
        let height = 96;
        let width = 128;
        let intrinsics = centered_intrinsics(height, width);
        let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
        let (depth, transform) = wavy_frame(height, width, 0);
        volume.integrate(&depth, [height, width], intrinsics, transform, None)?;
        volume.extract_dirty_block_meshes()
    };
    let meshes_one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()?
        .install(run)?;
    let meshes_eight = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()?
        .install(run)?;

    assert_eq!(meshes_one, meshes_eight);
    assert!(!meshes_one.is_empty());
    Ok(())
}

#[test]
fn integration_skips_world_coordinates_outside_the_sparse_index_range() -> Result<(), Box<dyn Error>>
{
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let intrinsics = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let camera_from_world = [
        1.0, 0.0, 0.0, -1.0e20, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    volume.integrate(&[1.0], [1, 1], intrinsics, camera_from_world, None)?;

    let mesh = volume.extract_mesh()?;
    assert!(mesh.vertices.is_empty());
    assert!(mesh.triangles.is_empty());
    Ok(())
}
