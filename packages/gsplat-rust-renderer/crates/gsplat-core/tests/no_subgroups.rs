//! The workgroup-memory scan and sort (devices without `SUBGROUP`, e.g. the `WebKit` WebGPU) render the
//! same pixels as the subgroup kernels.
#![cfg(feature = "testing")]
use glam::{Quat, Vec3};
use gsplat_core::test_utils as common;
use gsplat_core::{Camera, CameraModel, RenderOptions, Renderer, Splats, Target};

const SIZE: u32 = 256;

/// A dense pseudo-random cloud: millions of tile intersections through every sort and scan kernel.
/// Every splat has its own depth: with equal depths the order of the visible compaction (an atomic
/// counter, not deterministic) shows in the pixels of either variant (random f32 depths tie by the
/// thousand at 200k splats).
fn cloud(count: usize) -> Splats {
    let mut state = 0x9e37_79b9u32;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut transforms = Vec::with_capacity(count);
    let mut raw_opacities = Vec::with_capacity(count);
    let mut sh_coefficients = Vec::with_capacity(count);
    for i in 0..count {
        let z = (i as f32 + 0.5) / count as f32 * 2.0 - 1.0;
        let p = [next() * 2.0 - 1.0, next() * 2.0 - 1.0, z];
        let q = Quat::from_euler(
            glam::EulerRot::XYZ,
            next() * 6.3,
            next() * 6.3,
            next() * 6.3,
        );
        let s = -4.5 + next() * 2.0;
        transforms.push([
            p[0],
            p[1],
            p[2],
            q.w,
            q.x,
            q.y,
            q.z,
            s,
            s - next(),
            s + next() * 0.5,
        ]);
        raw_opacities.push(next() * 6.0 - 2.0);
        sh_coefficients.push([next() - 0.5, next() - 0.5, next() - 0.5]);
    }
    Splats {
        transforms,
        raw_opacities,
        sh_coefficients,
        sh_degree: 0,
        min_scale: None,
    }
}

fn render(subgroups: bool, splats: &Splats) -> (Vec<[f32; 4]>, u32) {
    let (device, queue) = &common::gpu_with(subgroups);
    let renderer = Renderer::new(device).expect("create renderer");
    assert_eq!(renderer.uses_subgroups(), subgroups);
    let scene = renderer.upload(splats).expect("upload cloud");
    let mut view = renderer.create_view(&scene, 1 << 16).expect("create view");
    let camera = Camera {
        model: CameraModel::Pinhole,
        position: Vec3::new(0.0, 0.0, -3.2),
        rotation: Quat::IDENTITY,
        fov_x: 0.9,
        fov_y: 0.9,
        center_uv: glam::Vec2::splat(0.5),
        size: glam::UVec2::splat(SIZE),
    };
    let target = common::upload(device, &vec![[0.0f32; 4]; (SIZE * SIZE) as usize]);
    for _ in 0..4 {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(
                queue,
                &mut encoder,
                &mut view,
                &camera,
                &RenderOptions::default(),
                Target::Float(target.clone()),
            )
            .expect("encode render");
        queue.submit([encoder.finish()]);
        let pixels = common::read::<[f32; 4]>(device, queue, &target, (SIZE * SIZE) as usize);
        let stats = view
            .poll_feedback()
            .expect("read feedback")
            .expect("feedback after a blocking read");
        if !stats.needs_rerender {
            return (pixels, stats.intersections);
        }
    }
    panic!("view never fit its intersections");
}

fn max_difference(a: &[[f32; 4]], b: &[[f32; 4]]) -> (usize, f32) {
    let differing = a.iter().zip(b).filter(|(a, b)| a != b).count();
    let max = a
        .iter()
        .zip(b)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
        .fold(0.0f32, f32::max);
    (differing, max)
}

#[test]
#[ignore = "integration: GPU"]
fn portable_scan_and_sort_render_the_subgroup_pixels() {
    let splats = cloud(230_000);
    let (with, intersections) = render(true, &splats);
    let (again, _) = render(true, &splats);
    let (without, portable_intersections) = render(false, &splats);
    let (without_again, _) = render(false, &splats);
    assert_eq!(intersections, portable_intersections);
    assert!(intersections > 1_000_000, "{intersections} intersections");
    let covered = with.iter().filter(|p| p[3] > 0.5).count();
    assert!(
        covered > (SIZE * SIZE / 4) as usize,
        "{covered} covered pixels"
    );
    let repeat = max_difference(&with, &again);
    let portable_repeat = max_difference(&without, &without_again);
    let variants = max_difference(&with, &without);
    println!(
        "differing pixels, max abs difference: subgroup repeat {repeat:?}, portable repeat {portable_repeat:?}, subgroup vs portable {variants:?}"
    );
    // Distinct depths make the frame deterministic: both variants must match bit for bit.
    assert_eq!(
        repeat.0 + portable_repeat.0,
        0,
        "renders of one variant differ"
    );
    assert_eq!(variants.0, 0, "pixels differ between the kernel variants");
}
