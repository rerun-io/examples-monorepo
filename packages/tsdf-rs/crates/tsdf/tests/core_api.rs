use std::collections::HashMap;
use std::error::Error;

use tsdf::{Camera, DepthFrame, DepthRange, DirtyMeshes, Mesh, TsdfError, TsdfVolume};

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
        for (key, mesh) in volume.extract_dirty_block_meshes() {
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
    let first = volume.extract_dirty_block_meshes();
    assert!(!first.is_empty());
    // Blocks that never held surface are omitted, not reported as empty.
    assert!(first.iter().all(|(_, mesh)| !mesh.triangles.is_empty()));
    assert!(volume.extract_dirty_block_meshes().is_empty());

    let (depth, camera) = wavy_frame(1);
    volume.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;
    assert!(!volume.extract_dirty_block_meshes().is_empty());
    Ok(())
}

#[test]
fn re_integrating_an_identical_frame_delivers_nothing() -> Result<(), Box<dyn Error>> {
    // A second identical observation leaves every running mean, and so every
    // mesh, bit for bit as it was; only the weights grow.
    let mut volume = TsdfVolume::new(0.04, 0.12, true)?;
    let (depth, camera) = wavy_frame(0);
    let color = vec![200_u8; HEIGHT * WIDTH * 3];
    let frame = DepthFrame::new(&depth, [HEIGHT, WIDTH]).with_color(&color);

    volume.integrate(&frame, &camera)?;
    assert!(!volume.extract_dirty_block_meshes().is_empty());
    volume.integrate(&frame, &camera)?;
    assert_eq!(volume.extract_dirty_block_meshes(), Vec::new());
    Ok(())
}

#[test]
fn an_unconfirmed_delivery_keeps_the_meshes_pending() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let (depth, camera) = wavy_frame(0);
    volume.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;

    let first = volume.dirty_block_meshes().meshes;
    assert!(!first.is_empty());
    let DirtyMeshes { meshes, delivery } = volume.dirty_block_meshes();
    assert_eq!(meshes, first);
    delivery.confirm();
    assert!(volume.dirty_block_meshes().meshes.is_empty());
    Ok(())
}

#[test]
fn dirty_block_meshes_are_invariant_to_rayon_thread_count() -> Result<(), Box<dyn Error>> {
    let run = || -> Result<Vec<([i64; 3], Mesh)>, TsdfError> {
        let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
        let (depth, camera) = wavy_frame(0);
        volume.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;
        Ok(volume.extract_dirty_block_meshes())
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

#[test]
fn the_block_budget_skips_new_blocks_and_keeps_fusing_old_ones() -> Result<(), Box<dyn Error>> {
    let (depth, camera) = wavy_frame(0);
    let frame = DepthFrame::new(&depth, [HEIGHT, WIDTH]);
    let mut free = TsdfVolume::new(0.04, 0.12, false)?;
    free.integrate(&frame, &camera)?;
    assert!(free.block_count() > 3 && !free.budget_reached());

    let mut capped = TsdfVolume::new(0.04, 0.12, false)?;
    capped.set_max_blocks(Some(3));
    capped.integrate(&frame, &camera)?;
    assert_eq!(capped.block_count(), 3);
    assert!(capped.budget_reached());
    let (depth, camera) = wavy_frame(1);
    capped.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;
    assert_eq!(capped.block_count(), 3);
    Ok(())
}

#[test]
fn eviction_frees_the_farthest_blocks_and_forgets_them() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    volume.set_max_blocks(Some(1_000));
    for frame in 0..3 {
        let (depth, camera) = wavy_frame(frame);
        volume.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;
    }
    let before = volume.block_count();
    let center = [0.0, 0.0, 2.0];
    let evicted = volume.evict_farthest(center, before / 2);
    assert_eq!(volume.block_count(), before / 2);
    assert_eq!(evicted.len(), before - before / 2);
    assert!(!volume.budget_reached());
    // Every kept block is at least as near as every evicted one.
    let distance = |key: &[i64; 3]| -> f32 {
        (0..3)
            .map(|axis| ((key[axis] as f32 + 0.5) * 0.64 - center[axis]).powi(2))
            .sum()
    };
    let nearest_evicted = evicted.iter().map(distance).fold(f32::INFINITY, f32::min);
    let meshes = volume.extract_dirty_block_meshes();
    assert!(meshes.iter().all(|(key, _)| !evicted.contains(key)));
    assert!(meshes
        .iter()
        .all(|(key, _)| distance(key) <= nearest_evicted));
    assert!(volume.evict_farthest(center, before).is_empty());
    Ok(())
}

#[test]
fn eviction_clears_the_budget_flag_even_when_nothing_is_evicted() -> Result<(), Box<dyn Error>> {
    let (depth, camera) = wavy_frame(0);
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    volume.set_max_blocks(Some(3));
    volume.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;
    assert!(volume.budget_reached());
    assert!(volume.evict_farthest([0.0, 0.0, 2.0], 3).is_empty());
    assert!(!volume.budget_reached());
    Ok(())
}

#[test]
fn min_weight_hides_cells_seen_once_from_live_meshes_only() -> Result<(), Box<dyn Error>> {
    let (depth, camera) = wavy_frame(0);
    let frame = DepthFrame::new(&depth, [HEIGHT, WIDTH]);
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    volume.set_min_weight(2.0);
    volume.integrate(&frame, &camera)?;
    assert!(volume
        .extract_dirty_block_meshes()
        .iter()
        .all(|(_, mesh)| mesh.triangles.is_empty()));
    assert!(!volume.extract_mesh()?.triangles.is_empty());
    volume.integrate(&frame, &camera)?;
    let live: usize = volume
        .extract_dirty_block_meshes()
        .iter()
        .map(|(_, mesh)| mesh.triangles.len())
        .sum();
    assert_eq!(live, volume.extract_mesh()?.triangles.len());
    Ok(())
}

#[test]
fn no_delivery_repeats_the_mesh_the_consumer_already_holds() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, true)?;
    let color: Vec<u8> = (0..HEIGHT * WIDTH * 3)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut held: HashMap<[i64; 3], Mesh> = HashMap::new();
    let (mut delivered, mut repeated) = (0, 0);
    for frame in 0..8 {
        let (depth, camera) = wavy_frame(frame);
        volume.integrate(
            &DepthFrame::new(&depth, [HEIGHT, WIDTH]).with_color(&color),
            &camera,
        )?;
        for (key, mesh) in volume.extract_dirty_block_meshes() {
            delivered += 1;
            if held.get(&key) == Some(&mesh) {
                repeated += 1;
            }
            held.insert(key, mesh);
        }
    }
    eprintln!(
        "wavy, 8 frames: {delivered} block meshes delivered, {repeated} identical to the held copy"
    );
    assert!(delivered > 0);
    assert_eq!(repeated, 0);
    Ok(())
}

#[test]
fn a_vanished_surface_is_delivered_empty_once() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let (depth, camera) = wavy_frame(0);
    let frame = DepthFrame::new(&depth, [HEIGHT, WIDTH]);
    volume.integrate(&frame, &camera)?;
    let surfaced: Vec<[i64; 3]> = volume
        .extract_dirty_block_meshes()
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    assert!(!surfaced.is_empty());

    // No voxel reaches this weight, so every live mesh empties.
    volume.set_min_weight(10.0);
    volume.integrate(&frame, &camera)?;
    let emptied = volume.extract_dirty_block_meshes();
    assert_eq!(
        emptied.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        surfaced
    );
    assert!(emptied.iter().all(|(_, mesh)| mesh.triangles.is_empty()));
    volume.integrate(&frame, &camera)?;
    assert_eq!(volume.extract_dirty_block_meshes(), Vec::new());
    Ok(())
}

/// A sphere of radius 1.7 around the camera, which turns by `yaw` radians.
fn sphere_frame(yaw: f32) -> (Vec<f32>, Camera) {
    let camera = centered_camera(HEIGHT, WIDTH, Camera::IDENTITY);
    let [focal, principal_x, principal_y] = [
        camera.intrinsics[0],
        camera.intrinsics[2],
        camera.intrinsics[5],
    ];
    let depth = (0..HEIGHT * WIDTH)
        .map(|index| {
            let x = ((index % WIDTH) as f32 - principal_x) / focal;
            let y = ((index / WIDTH) as f32 - principal_y) / focal;
            1.7 / (x * x + y * y + 1.0).sqrt()
        })
        .collect();
    let (sin, cos) = yaw.sin_cos();
    #[rustfmt::skip]
    let camera_from_world = [
        cos, 0.0, -sin, 0.0,
        0.0, 1.0, 0.0, 0.0,
        sin, 0.0, cos, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    (depth, centered_camera(HEIGHT, WIDTH, camera_from_world))
}

/// The triangles of the latest mesh per block, as [`triangle_multiset`].
fn union_multiset(latest: &HashMap<[i64; 3], Mesh>) -> Vec<[[u32; 3]; 3]> {
    let mut triangles: Vec<[[u32; 3]; 3]> = latest.values().flat_map(triangle_multiset).collect();
    triangles.sort_unstable();
    triangles
}

#[test]
fn after_eviction_the_held_union_still_equals_the_full_mesh() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let mut held: HashMap<[i64; 3], Mesh> = HashMap::new();
    for frame in 0..30 {
        let (depth, camera) = sphere_frame(frame as f32 * 0.03);
        volume.integrate(&DepthFrame::new(&depth, [HEIGHT, WIDTH]), &camera)?;
        held.extend(volume.extract_dirty_block_meshes());
    }
    let count = volume.block_count();
    let evicted = volume.evict_farthest([0.0, 0.0, 1.7], count * 2 / 3);
    assert!(!evicted.is_empty());
    // The consumer clears evicted blocks; the kept neighbors' meshes lose
    // the cells that read the evicted voxels and come again.
    for key in &evicted {
        held.remove(key);
    }
    held.extend(volume.extract_dirty_block_meshes());
    let (union, full) = (
        union_multiset(&held),
        triangle_multiset(&volume.extract_mesh()?),
    );
    assert!(
        union == full,
        "held union has {} triangles, extract_mesh {}",
        union.len(),
        full.len()
    );
    Ok(())
}

#[test]
fn a_block_recreated_after_eviction_is_delivered_again() -> Result<(), Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let (depth, camera) = wavy_frame(0);
    let frame = DepthFrame::new(&depth, [HEIGHT, WIDTH]);
    volume.integrate(&frame, &camera)?;
    let first = volume.extract_dirty_block_meshes();
    assert!(!first.is_empty());

    volume.evict_farthest([0.0, 0.0, 2.0], 0);
    assert_eq!(volume.block_count(), 0);
    // The same frame rebuilds the same blocks with the same meshes; the
    // consumer may have cleared them on eviction, so they come again.
    volume.integrate(&frame, &camera)?;
    assert_eq!(volume.extract_dirty_block_meshes(), first);
    Ok(())
}
