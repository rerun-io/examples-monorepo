use std::collections::HashMap;
use std::error::Error;

use tsdf::{Camera, DepthFrame, DepthRange, Mesh, TsdfError, TsdfVolume};

/// Per position (f32 bits), the (color, normal bits) of each full-mesh vertex there.
type AttributesByPosition = HashMap<[u32; 3], Vec<([u8; 3], [u32; 3])>>;

const HEIGHT: usize = 96;
const WIDTH: usize = 128;

fn centered_camera(height: usize, width: usize, camera_from_world: [f32; 16]) -> Camera {
    Camera::pinhole(
        [
            100.0,
            100.0,
            width as f32 / 2.0 - 0.5,
            height as f32 / 2.0 - 0.5,
        ],
        camera_from_world,
    )
}

/// A wall at depth ~2 rippling along x; the camera moves along x each frame,
/// so new blocks appear beside blocks that are updated at the same time.
fn wavy_frame(frame: usize) -> (Vec<f32>, Camera) {
    let depth = (0..HEIGHT * WIDTH)
        .map(|index| {
            let u = (index % WIDTH) as f32;
            2.0 + 0.03 * frame as f32 + 0.1 * (u / 9.0).sin()
        })
        .collect();
    let mut camera_from_world = Camera::IDENTITY;
    camera_from_world[3] = -0.4 * frame as f32;
    (depth, centered_camera(HEIGHT, WIDTH, camera_from_world))
}

fn plane_mesh() -> Result<Mesh, TsdfError> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let depth = vec![2.0_f32; HEIGHT * WIDTH];
    let camera = centered_camera(HEIGHT, WIDTH, Camera::IDENTITY);
    volume.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;
    volume.extract_mesh()
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

#[test]
fn integration_reports_slice_lengths_against_the_image_size() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let camera = centered_camera(2, 2, Camera::IDENTITY);

    let short_depth = volume.integrate(&DepthFrame::new(&[1.0; 3], [2, 2]), &camera);
    assert_eq!(
        short_depth,
        Err(TsdfError::ShapeMismatch {
            name: "depth",
            expected: 4,
            got: 3
        })
    );
    let short_color = volume.integrate(
        &DepthFrame::new(&[1.0; 4], [2, 2]).with_color(&[0; 11]),
        &camera,
    );
    assert_eq!(
        short_color,
        Err(TsdfError::ShapeMismatch {
            name: "color",
            expected: 12,
            got: 11
        })
    );
    Ok(())
}

#[test]
fn invalid_depth_ranges_are_refused() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let camera = centered_camera(2, 2, Camera::IDENTITY);
    for (min, max, name) in [
        (-0.1, 1.0, "min_depth"),
        (f32::NAN, 1.0, "min_depth"),
        (2.0, 1.0, "max_depth"),
        (0.0, f32::NAN, "max_depth"),
    ] {
        let frame = DepthFrame::new(&[1.0; 4], [2, 2]).with_depth_range(DepthRange { min, max });
        assert!(matches!(
            volume.integrate(&frame, &camera),
            Err(TsdfError::InvalidParameter { name: got, .. }) if got == name
        ));
    }
    Ok(())
}

#[test]
fn depths_outside_the_range_are_not_integrated() -> Result<(), Box<dyn Error>> {
    // A near wall (columns < 64) and a far wall; the range keeps only the near one.
    let depth: Vec<f32> = (0..HEIGHT * WIDTH)
        .map(|index| if index % WIDTH < 64 { 1.5 } else { 6.0 })
        .collect();
    let masked: Vec<f32> = depth
        .iter()
        .map(|&value| if value <= 3.0 { value } else { 0.0 })
        .collect();
    let camera = centered_camera(HEIGHT, WIDTH, Camera::IDENTITY);
    let range = DepthRange { min: 0.2, max: 3.0 };

    let mut ranged = TsdfVolume::new(0.04, 0.12, false)?;
    ranged.integrate(
        &DepthFrame::new(&depth, [HEIGHT, WIDTH]).with_depth_range(range),
        &camera,
    )?;
    let mut prefiltered = TsdfVolume::new(0.04, 0.12, false)?;
    prefiltered.integrate(&DepthFrame::new(&masked, [HEIGHT, WIDTH]), &camera)?;

    let mesh = ranged.extract_mesh()?;
    assert!(!mesh.vertices.is_empty());
    assert!(mesh.vertices.iter().all(|vertex| vertex[2] < 2.0));
    assert_eq!(mesh, prefiltered.extract_mesh()?);
    Ok(())
}

#[test]
fn mesh_bytes_are_invariant_to_rayon_thread_count() -> Result<(), Box<dyn Error>> {
    let mesh_one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()?
        .install(plane_mesh)?;
    let mesh_eight = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()?
        .install(plane_mesh)?;

    assert_eq!(mesh_one, mesh_eight);
    assert!(!mesh_one.vertices.is_empty());
    Ok(())
}

#[test]
fn plane_normals_are_unit_and_face_the_camera() -> Result<(), Box<dyn Error>> {
    let mesh = plane_mesh()?;

    assert_eq!(mesh.normals.len(), mesh.vertices.len());
    for normal in &mesh.normals {
        let length = normal.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-5);
    }
    // The camera at the origin looks down +Z at the wall: interior normals point back at it.
    let facing = mesh
        .normals
        .iter()
        .filter(|normal| normal[2] < -0.99)
        .count();
    assert!(
        facing * 10 > mesh.normals.len() * 9,
        "{facing} of {}",
        mesh.normals.len()
    );
    // Counter-clockwise winding seen from outside: face normals agree with vertex normals.
    for triangle in &mesh.triangles {
        let [a, b, c] = triangle.map(|index| mesh.vertices[index as usize]);
        let face_z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        assert!(face_z <= 0.0);
    }
    Ok(())
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
    let camera = centered_camera(height, width, Camera::IDENTITY);
    let initial_depth = vec![INITIAL_DEPTH; height * width];
    let shifted_depth = vec![SHIFTED_DEPTH; height * width];

    for _ in 0..INITIAL_OBSERVATIONS {
        volume.integrate(&DepthFrame::new(&initial_depth, [height, width]), &camera)?;
    }
    for _ in 0..SHIFTED_OBSERVATIONS {
        volume.integrate(&DepthFrame::new(&shifted_depth, [height, width]), &camera)?;
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
fn dirty_block_union_matches_full_extraction_after_every_frame() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, true)?;
    let mut latest: HashMap<[i64; 3], Mesh> = HashMap::new();
    let color: Vec<u8> = (0..HEIGHT * WIDTH * 3)
        .map(|index| (index % 251) as u8)
        .collect();

    for frame in 0..3 {
        let (depth, camera) = wavy_frame(frame);
        volume.integrate(
            &DepthFrame::new(&depth, [HEIGHT, WIDTH]).with_color(&color),
            &camera,
        )?;
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

        // Colors and normals are functions of the vertex's edge, so block
        // meshes agree with the full mesh wherever positions coincide (seams
        // included). Distinct edges can meet at one point; compare against
        // every full-mesh vertex there.
        let full_colors = full.colors.as_ref().ok_or("full mesh missing colors")?;
        let mut full_attributes = AttributesByPosition::new();
        for ((vertex, color), normal) in full.vertices.iter().zip(full_colors).zip(&full.normals) {
            full_attributes
                .entry(vertex.map(f32::to_bits))
                .or_default()
                .push((*color, normal.map(f32::to_bits)));
        }
        for mesh in latest.values() {
            let colors = mesh.colors.as_ref().ok_or("block mesh missing colors")?;
            for ((vertex, color), normal) in mesh.vertices.iter().zip(colors).zip(&mesh.normals) {
                let expected = full_attributes
                    .get(&vertex.map(f32::to_bits))
                    .ok_or("block vertex missing from the full mesh")?;
                assert!(
                    expected.contains(&(*color, normal.map(f32::to_bits))),
                    "block vertex color or normal diverged from the full mesh at frame {frame}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn extract_dirty_block_meshes_drains_the_dirty_set() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let (depth, camera) = wavy_frame(0);
    let frame = DepthFrame::new(&depth, [HEIGHT, WIDTH]);

    volume.integrate(&frame, &camera)?;
    let first = volume.extract_dirty_block_meshes()?;
    assert!(!first.is_empty());
    // Blocks that never held surface are omitted, not reported as empty.
    assert!(first.iter().all(|(_, mesh)| !mesh.triangles.is_empty()));
    assert!(volume.extract_dirty_block_meshes()?.is_empty());

    volume.integrate(&frame, &camera)?;
    assert!(!volume.extract_dirty_block_meshes()?.is_empty());
    Ok(())
}

#[test]
fn an_uncommitted_extraction_keeps_the_tracker_and_a_stale_one_is_refused(
) -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let (depth, camera) = wavy_frame(0);
    let frame = DepthFrame::new(&depth, [HEIGHT, WIDTH]);
    volume.integrate(&frame, &camera)?;

    let first = volume.dirty_block_meshes()?;
    assert_eq!(volume.dirty_block_meshes()?.meshes(), first.meshes());

    volume.integrate(&frame, &camera)?;
    assert_eq!(volume.commit(&first), Err(TsdfError::StaleDirtyMeshes));
    let current = volume.dirty_block_meshes()?;
    volume.commit(&current)?;
    assert!(volume.dirty_block_meshes()?.meshes().is_empty());
    Ok(())
}

#[test]
fn dirty_block_meshes_are_invariant_to_rayon_thread_count() -> Result<(), Box<dyn Error>> {
    let run = || -> Result<Vec<([i64; 3], Mesh)>, TsdfError> {
        let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
        let (depth, camera) = wavy_frame(0);
        volume.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;
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
    let mut camera_from_world = Camera::IDENTITY;
    camera_from_world[3] = -1.0e20;
    let camera = Camera::pinhole([1.0, 1.0, 0.0, 0.0], camera_from_world);

    volume.integrate(&DepthFrame::new(&[1.0], [1, 1]), &camera)?;

    let mesh = volume.extract_mesh()?;
    assert!(mesh.vertices.is_empty());
    assert!(mesh.triangles.is_empty());
    Ok(())
}

#[test]
fn singular_intrinsics_are_refused() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let camera = Camera::pinhole([0.0, 100.0, 0.5, 0.5], Camera::IDENTITY);
    assert_eq!(
        volume.integrate(&DepthFrame::new(&[1.0; 4], [2, 2]), &camera),
        Err(TsdfError::SingularMatrix { name: "intrinsics" })
    );
    Ok(())
}
