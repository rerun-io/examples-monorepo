//! Hexel GPU contracts against an independent f64 CPU forward model.
//!
//! The reference intersects each pixel ray with the hexel plane by solving
//! `c + u a + v b = t r` (Cramer's rule), not through the GPU's plane-to-pixel map,
//! and composites in centre-depth order without tiles.
use crate::test_utils::{gpu, read, upload};
use crate::{
    Camera, CameraModel, FrameStats, HexelVariant, Hexels, RenderOptions, Renderer, Target,
    ViewState,
};
use glam::{DAffine3, DMat3, DQuat, DVec3, Quat, UVec2, Vec2, Vec3};

const NEAR: f64 = 0.2;

struct Image {
    rgba: Vec<[f64; 4]>,
    /// Expected and median depth.
    depth: Vec<[f64; 2]>,
    normal: Vec<[f64; 3]>,
}

struct Projected {
    center: DVec3,
    a: DVec3,
    b: DVec3,
    normal: DVec3,
    opacity: f64,
    sharpness: f64,
    color: [DVec3; 3],
}

fn sh_basis(d: DVec3) -> [f64; 15] {
    let (x, y, z) = (d.x, d.y, d.z);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let c1 = 0.488_602_511_902_919_9;
    [
        -c1 * y,
        c1 * z,
        -c1 * x,
        1.092_548_430_592_079_2 * x * y,
        -1.092_548_430_592_079_2 * y * z,
        0.315_391_565_252_520_05 * (2.0 * zz - xx - yy),
        -1.092_548_430_592_079_2 * x * z,
        0.546_274_215_296_039_6 * (xx - yy),
        -0.590_043_589_926_643_5 * y * (3.0 * xx - yy),
        2.890_611_442_640_554 * x * y * z,
        -0.457_045_799_464_465_8 * y * (4.0 * zz - xx - yy),
        0.373_176_332_590_115_4 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        -0.457_045_799_464_465_8 * x * (4.0 * zz - xx - yy),
        1.445_305_721_320_277 * z * (xx - yy),
        -0.590_043_589_926_643_5 * x * (xx - 3.0 * yy),
    ]
}

fn reference(hexels: &Hexels, camera: &Camera, options: &RenderOptions) -> Image {
    let world_from_local =
        DAffine3::from_mat4(glam::Mat4::from(options.world_from_local).as_dmat4());
    let world_from_camera = DAffine3::from_rotation_translation(
        DQuat::from_array(camera.rotation.to_array().map(f64::from)),
        camera.position.as_dvec3(),
    );
    let view_from_local = world_from_camera.inverse() * world_from_local;
    let eye_local = world_from_local
        .inverse()
        .transform_point3(camera.position.as_dvec3());
    let stride = hexels.variant.stride();
    let mut projected: Vec<Projected> = hexels
        .params
        .chunks_exact(stride)
        .filter_map(|h| {
            let h: Vec<f64> = h.iter().map(|&x| f64::from(x)).collect();
            let v3 = |i: usize| DVec3::new(h[i], h[i + 1], h[i + 2]);
            let rotation = DMat3::from_quat(DQuat::from_xyzw(h[4], h[5], h[6], h[3]).normalize());
            let scale = f64::from(options.splat_scale);
            let a = view_from_local.transform_vector3(rotation.x_axis * h[7] * scale);
            let b = view_from_local.transform_vector3(rotation.y_axis * h[8] * scale);
            let center = view_from_local.transform_point3(v3(0));
            if center.z <= NEAR || h[9] < 1.0 / 255.0 {
                return None;
            }
            let direction = (v3(0) - eye_local).normalize();
            let view_term = match hexels.variant {
                HexelVariant::V1 | HexelVariant::Image => DVec3::ZERO,
                HexelVariant::V2 => {
                    DMat3::from_cols(v3(20), v3(23), v3(26)).transpose() * direction
                }
                HexelVariant::V3 => sh_basis(direction)
                    .iter()
                    .enumerate()
                    .map(|(k, y)| v3(20 + 3 * k) * *y)
                    .sum(),
            };
            let world_a = world_from_local.transform_vector3(rotation.x_axis);
            let world_b = world_from_local.transform_vector3(rotation.y_axis);
            let mut normal = world_a.cross(world_b).normalize();
            if normal.dot(world_from_local.transform_point3(v3(0)) - camera.position.as_dvec3())
                > 0.0
            {
                normal = -normal;
            }
            Some(Projected {
                center,
                a,
                b,
                normal,
                opacity: h[9],
                sharpness: h[10].max(1e-4),
                color: [v3(11) + view_term, v3(14), v3(17)],
            })
        })
        .collect();
    projected.sort_by(|p, q| p.center.z.total_cmp(&q.center.z));
    let size = camera.size;
    let focal = camera.focal().as_dvec2();
    let principal = (camera.center_uv * size.as_vec2()).as_dvec2();
    let background = options.background.as_dvec3();
    let mut image = Image {
        rgba: Vec::new(),
        depth: Vec::new(),
        normal: Vec::new(),
    };
    for y in 0..size.y {
        for x in 0..size.x {
            let pixel = glam::DVec2::new(f64::from(x), f64::from(y)) + 0.5;
            let ray = ((pixel - principal) / focal).extend(1.0);
            let (mut transmittance, mut color, mut depth, mut median, mut normal) =
                (1.0, DVec3::ZERO, 0.0, 0.0, DVec3::ZERO);
            for p in &projected {
                // c + u a + v b - t r = 0.
                let m = DMat3::from_cols(p.a, p.b, -ray);
                let det = m.determinant();
                if det == 0.0 {
                    continue;
                }
                let solve = |column: usize| {
                    let mut replaced = m;
                    *replaced.col_mut(column) = -p.center;
                    replaced.determinant() / det
                };
                let (u, v, t) = (solve(0), solve(1), solve(2));
                if t < NEAR {
                    continue;
                }
                let k = 3f64.sqrt() * 0.5;
                let sdf = v
                    .abs()
                    .max((k * u + 0.5 * v).abs())
                    .max((k * u - 0.5 * v).abs())
                    - 1.0;
                let alpha = (p.opacity / (1.0 + (sdf * p.sharpness).exp())).min(0.99);
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let next = transmittance * (1.0 - alpha);
                if next < 1e-4 {
                    break;
                }
                let w = alpha * transmittance;
                color += (p.color[0] + u * p.color[1] + v * p.color[2]) * w;
                if transmittance > 0.5 {
                    median = t;
                }
                depth += t * w;
                normal += p.normal * w;
                transmittance = next;
            }
            let alpha = 1.0 - transmittance;
            let c = color + transmittance * background;
            image.rgba.push([c.x, c.y, c.z, alpha]);
            image.depth.push([depth / alpha.max(1e-8), median]);
            image.normal.push((normal / alpha.max(1e-8)).to_array());
        }
    }
    image
}

struct Gpu {
    rgba: Vec<[f32; 4]>,
    depth: Vec<[f32; 2]>,
    normal: Vec<[f32; 4]>,
}

/// Render `target`, again while the view grows its intersection buffer (the first frame holds one).
fn render_fitted(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer,
    view: &mut ViewState,
    camera: &Camera,
    options: &RenderOptions,
    target: &Target,
) -> FrameStats {
    loop {
        let mut encoder = device.create_command_encoder(&Default::default());
        let frame = renderer.render(queue, &mut encoder, view, camera, options, target.clone());
        frame.unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let stats = view.poll_feedback().unwrap().unwrap();
        if !stats.needs_rerender {
            return stats;
        }
    }
}

fn render(hexels: &Hexels, camera: &Camera, options: &RenderOptions) -> Gpu {
    let (device, queue) = &gpu();
    let renderer = Renderer::new(device).unwrap();
    let scene = renderer.upload_hexels(hexels).unwrap();
    let mut view = renderer.create_view(&scene, 1).unwrap();
    let pixels = (camera.size.x * camera.size.y) as usize;
    let color = upload(device, &vec![[0.0f32; 4]; pixels]);
    let float = upload(device, &vec![[0.0f32; 4]; pixels]);
    let depth = upload(device, &vec![[0.0f32; 2]; pixels]);
    let normal = upload(device, &vec![[0.0f32; 4]; pixels]);
    for target in [
        Target::Surface {
            color: color.clone(),
            depth: depth.clone(),
            normal: normal.clone(),
        },
        Target::Float(float.clone()),
    ] {
        render_fitted(
            device, queue, &renderer, &mut view, camera, options, &target,
        );
    }
    let rgba = read::<[f32; 4]>(device, queue, &color, pixels);
    let rgb_only = read::<[f32; 4]>(device, queue, &float, pixels);
    for (surface, plain) in rgba.iter().zip(&rgb_only) {
        for (s, p) in surface.iter().zip(plain) {
            assert!(
                (s - p).abs() <= 1e-6,
                "surface colour {s} != float colour {p}"
            );
        }
    }
    Gpu {
        rgba,
        depth: read(device, queue, &depth, pixels),
        normal: read(device, queue, &normal, pixels),
    }
}

/// PSNR over RGBA with peak 1; identical images report infinity.
fn psnr(gpu: &[[f32; 4]], cpu: &[[f64; 4]]) -> f64 {
    let squared: f64 = gpu
        .iter()
        .zip(cpu)
        .flat_map(|(g, c)| g.iter().zip(c).map(|(g, c)| (f64::from(*g) - c).powi(2)))
        .sum();
    -10.0 * (squared / (gpu.len() * 4) as f64).log10()
}

fn check(hexels: &Hexels, camera: &Camera, options: &RenderOptions) -> f64 {
    let gpu = render(hexels, camera, options);
    let cpu = reference(hexels, camera, options);
    let psnr = psnr(&gpu.rgba, &cpu.rgba);
    let covered = cpu.rgba.iter().filter(|p| p[3] > 1e-3).count();
    println!("{covered} covered pixels, GPU vs CPU {psnr:.1} dB");
    assert!(covered > 20, "scene does not cover the image");
    assert!(psnr >= 60.0, "GPU vs CPU PSNR {psnr:.2} dB");
    for i in 0..gpu.rgba.len() {
        let alpha = cpu.rgba[i][3];
        if alpha < 0.05 || (f64::from(gpu.rgba[i][3]) - alpha).abs() > 1e-4 {
            continue; // Depth and normal are undefined or dominated by an alpha difference.
        }
        let [expected, median] = cpu.depth[i];
        let relative = |gpu: f32, cpu: f64| (f64::from(gpu) - cpu).abs() / cpu.abs().max(1e-6);
        assert!(
            relative(gpu.depth[i][0], expected) <= 1e-4,
            "pixel {i}: expected depth {:?} vs {expected}",
            gpu.depth[i]
        );
        if median > 0.0 {
            assert!(
                relative(gpu.depth[i][1], median) <= 1e-4,
                "pixel {i}: median depth {:?} vs {median}",
                gpu.depth[i]
            );
        }
        let error =
            DVec3::from_array(cpu.normal[i]) - Vec3::from_slice(&gpu.normal[i][..3]).as_dvec3();
        assert!(
            error.length() <= 1e-4,
            "pixel {i}: normal {:?} vs {:?}",
            gpu.normal[i],
            cpu.normal[i]
        );
    }
    psnr
}

/// One hexel row in contract order; view coefficients follow when the variant has them.
fn hexel(
    center: [f32; 3],
    rotation: Quat,
    scale: [f32; 2],
    opacity: f32,
    sharpness: f32,
    colors: [[f32; 3]; 3],
) -> Vec<f32> {
    let mut row = center.to_vec();
    row.extend([rotation.w, rotation.x, rotation.y, rotation.z]);
    row.extend(scale);
    row.extend([opacity, sharpness]);
    row.extend(colors.into_iter().flatten());
    row
}

fn camera(size: UVec2) -> Camera {
    Camera {
        model: CameraModel::Pinhole,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        fov_x: 1.0,
        fov_y: 2.0 * ((0.5f64).tan() * f64::from(size.y) / f64::from(size.x)).atan(),
        center_uv: Vec2::splat(0.5),
        size,
    }
}

/// Deterministic values in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn random_scene(variant: HexelVariant, count: usize, seed: u64) -> Hexels {
    let mut rng = Lcg(seed);
    let mut params = Vec::new();
    for _ in 0..count {
        let center = [rng.next() * 1.2, rng.next() * 0.9, 3.0 + rng.next() * 1.5];
        let rotation = Quat::from_euler(
            glam::EulerRot::XYZ,
            rng.next() * 1.3,
            rng.next() * 1.3,
            rng.next() * 3.0,
        );
        let scale = [0.15 + 0.1 * rng.next(), 0.15 + 0.1 * rng.next()];
        let colors = [
            [
                0.5 + 0.4 * rng.next(),
                0.5 + 0.4 * rng.next(),
                0.5 + 0.4 * rng.next(),
            ],
            [0.3 * rng.next(), 0.3 * rng.next(), 0.3 * rng.next()],
            [0.3 * rng.next(), 0.3 * rng.next(), 0.3 * rng.next()],
        ];
        params.extend(hexel(
            center,
            rotation,
            scale,
            0.6 + 0.35 * rng.next(),
            12.0 + 8.0 * rng.next(),
            colors,
        ));
        params.extend((20..variant.stride()).map(|_| 0.2 * rng.next()));
    }
    Hexels { variant, params }
}

#[test]
#[ignore = "integration: GPU"]
fn single_frontal_hexel_matches_the_cpu_model_with_gradients() {
    let colors = [[0.6, 0.3, 0.2], [0.2, -0.1, 0.05], [-0.05, 0.15, 0.1]];
    let hexels = Hexels {
        variant: HexelVariant::V1,
        params: hexel(
            [0.1, -0.05, 2.0],
            Quat::from_rotation_z(0.4),
            [0.5, 0.3],
            0.9,
            20.0,
            colors,
        ),
    };
    let options = RenderOptions {
        background: Vec3::new(0.1, 0.2, 0.3),
        ..Default::default()
    };
    check(&hexels, &camera(UVec2::new(80, 64)), &options);
}

#[test]
#[ignore = "integration: GPU"]
fn overlapping_hexels_composite_in_centre_depth_order() {
    let mut params = Vec::new();
    for (i, z) in [2.0f32, 2.4, 2.8, 3.3].into_iter().enumerate() {
        let offset = i as f32 * 0.12 - 0.18;
        params.extend(hexel(
            [offset, -offset * 0.5, z],
            Quat::from_rotation_y(0.3 * i as f32 - 0.4) * Quat::from_rotation_z(0.2 * i as f32),
            [0.45, 0.35],
            [0.7, 0.5, 0.97, 0.8][i],
            [8.0, 30.0, 15.0, 60.0][i],
            [
                [0.2 * i as f32, 0.5, 0.9 - 0.2 * i as f32],
                [0.1, 0.0, -0.1],
                [0.0, 0.1, 0.05],
            ],
        ));
    }
    let hexels = Hexels {
        variant: HexelVariant::V1,
        params,
    };
    let mut camera = camera(UVec2::new(96, 72));
    camera.center_uv = Vec2::new(0.47, 0.53);
    check(&hexels, &camera, &RenderOptions::default());
}

#[test]
#[ignore = "integration: GPU"]
fn small_hexels_on_tile_corners_keep_every_pixel() {
    // Pixel (16, 16) is the corner of four tiles; its centre is the ray through (16.5, 16.5).
    let camera = camera(UVec2::new(64, 48));
    let focal = camera.focal();
    let mut params = Vec::new();
    for (i, pixel) in [
        Vec2::new(16.0, 16.0),
        Vec2::new(32.0, 32.0),
        Vec2::new(47.5, 16.0),
    ]
    .into_iter()
    .enumerate()
    {
        let z = 2.0 + i as f32 * 0.3;
        let center = ((pixel - camera.center_uv * camera.size.as_vec2()) / focal * z).extend(z);
        params.extend(hexel(
            center.to_array(),
            Quat::from_rotation_z(0.3),
            [0.06, 0.05],
            0.95,
            25.0,
            [[0.9, 0.4, 0.1], [0.0; 3], [0.0; 3]],
        ));
    }
    check(
        &Hexels {
            variant: HexelVariant::V1,
            params,
        },
        &camera,
        &RenderOptions::default(),
    );
}

#[test]
#[ignore = "integration: GPU"]
fn steep_grazing_angles_match_the_cpu_model() {
    let mut params = Vec::new();
    for (i, angle) in [1.2f32, 1.4, 1.5, 1.54].into_iter().enumerate() {
        let x = i as f32 * 0.35 - 0.5;
        let axis = if i % 2 == 0 {
            Quat::from_rotation_y(angle)
        } else {
            Quat::from_rotation_x(angle)
        };
        params.extend(hexel(
            [x, 0.1 - 0.1 * i as f32, 2.5],
            axis,
            [0.4, 0.3],
            0.9,
            15.0,
            [[0.8, 0.6, 0.2], [0.1, 0.2, 0.0], [0.0, -0.2, 0.1]],
        ));
    }
    check(
        &Hexels {
            variant: HexelVariant::V1,
            params,
        },
        &camera(UVec2::new(96, 64)),
        &RenderOptions::default(),
    );
}

#[test]
#[ignore = "integration: GPU"]
fn tilted_hexels_map_only_the_tiles_they_reach() {
    // Tilted 1.2 rad about a diagonal, the hexel is a slanted band across its tile rectangle; the
    // tile lists keep about the tiles where the f64 model reaches alpha >= 1/255, not the rectangle.
    let camera = camera(UVec2::new(256, 192));
    let hexels = Hexels {
        variant: HexelVariant::V1,
        params: hexel(
            [0.0, 0.0, 3.0],
            Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize_or_zero(), 1.2),
            [1.2, 0.5],
            0.9,
            20.0,
            [[0.7, 0.5, 0.2], [0.0; 3], [0.0; 3]],
        ),
    };
    let options = RenderOptions::default();
    let (device, queue) = &gpu();
    let renderer = Renderer::new(device).unwrap();
    let scene = renderer.upload_hexels(&hexels).unwrap();
    let mut view = renderer.create_view(&scene, 1).unwrap();
    let pixels = (camera.size.x * camera.size.y) as usize;
    let target = Target::Float(upload(device, &vec![[0.0f32; 4]; pixels]));
    let stats = render_fitted(
        device, queue, &renderer, &mut view, &camera, &options, &target,
    );
    let cpu = reference(&hexels, &camera, &options);
    let tiles_x = camera.size.x.div_ceil(16);
    let reached: std::collections::HashSet<u32> = (0..camera.size.x * camera.size.y)
        .filter(|&i| cpu.rgba[i as usize][3] >= 1.0 / 255.0)
        .map(|i| i % camera.size.x / 16 + i / camera.size.x / 16 * tiles_x)
        .collect();
    assert!(
        stats.intersections as usize <= reached.len() + reached.len() / 4,
        "{} intersections for {} reached tiles",
        stats.intersections,
        reached.len()
    );
    check(&hexels, &camera, &options);
}

#[test]
#[ignore = "integration: GPU"]
fn random_scenes_match_for_every_variant() {
    for (variant, seed) in [
        (HexelVariant::V1, 1),
        (HexelVariant::V2, 2),
        (HexelVariant::V3, 3),
    ] {
        let hexels = random_scene(variant, 150, seed);
        let mut camera = camera(UVec2::new(120, 90));
        camera.position = Vec3::new(0.3, -0.2, -0.4);
        camera.rotation = Quat::from_rotation_y(-0.08) * Quat::from_rotation_x(0.05);
        let psnr = check(
            &hexels,
            &camera,
            &RenderOptions {
                background: Vec3::splat(0.25),
                ..Default::default()
            },
        );
        println!("{variant:?}: {psnr:.1} dB");
    }
}

#[test]
#[ignore = "integration: GPU"]
fn instance_transform_and_scale_modifier_move_hexels_and_view_terms() {
    let hexels = random_scene(HexelVariant::V3, 60, 7);
    let options = RenderOptions {
        world_from_local: glam::Affine3A::from_scale_rotation_translation(
            Vec3::splat(1.3),
            Quat::from_rotation_y(0.2),
            Vec3::new(0.1, 0.0, 0.6),
        ),
        splat_scale: 0.8,
        ..Default::default()
    };
    check(&hexels, &camera(UVec2::new(96, 72)), &options);
}

#[test]
#[ignore = "integration: GPU"]
fn near_plane_culls_centres_and_intersections() {
    // Upstream drops hexels whose centre is within 0.2 of the camera.
    let near = hexel(
        [0.0, 0.0, 0.15],
        Quat::IDENTITY,
        [0.5, 0.5],
        0.9,
        20.0,
        [[1.0, 0.0, 0.0], [0.0; 3], [0.0; 3]],
    );
    let far = hexel(
        [0.0, 0.0, 2.0],
        Quat::IDENTITY,
        [0.8, 0.8],
        0.9,
        20.0,
        [[0.0, 0.0, 1.0], [0.0; 3], [0.0; 3]],
    );
    let hexels = Hexels {
        variant: HexelVariant::V1,
        params: [near, far].concat(),
    };
    let camera = camera(UVec2::new(64, 48));
    let gpu = render(&hexels, &camera, &RenderOptions::default());
    let center = gpu.rgba[24 * 64 + 32];
    assert!(center[0] < 1e-6 && center[2] > 0.5, "{center:?}");
    check(&hexels, &camera, &RenderOptions::default());
}

#[test]
#[ignore = "integration: GPU"]
fn hexels_reject_non_pinhole_lenses_and_images_reject_surface_targets() {
    let (device, queue) = &gpu();
    let renderer = Renderer::new(device).unwrap();
    let mut camera = camera(UVec2::new(32, 32));
    let buffer = |bytes: usize| upload(device, &vec![0u8; bytes]);
    let surface = Target::Surface {
        color: buffer(32 * 32 * 16),
        depth: buffer(32 * 32 * 8),
        normal: buffer(32 * 32 * 16),
    };
    let image = Hexels {
        variant: HexelVariant::Image,
        params: vec![
            16.0, 16.0, 4.0, 4.0, 0.0, 10.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    };
    let image = renderer.upload_hexels(&image).unwrap();
    let mut view = renderer.create_view(&image, 64).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let options = RenderOptions::default();
    assert!(
        renderer
            .render(
                queue,
                &mut encoder,
                &mut view,
                &camera,
                &options,
                surface.clone()
            )
            .is_err()
    );
    let hexels = renderer
        .upload_hexels(&random_scene(HexelVariant::V1, 4, 9))
        .unwrap();
    let mut view = renderer.create_view(&hexels, 64).unwrap();
    camera.model = CameraModel::KannalaBrandt4 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        k4: 0.0,
    };
    assert!(
        renderer
            .render(queue, &mut encoder, &mut view, &camera, &options, surface)
            .is_err()
    );
}

/// Three 0.3-alpha layers: transmittance 1, 0.7, 0.49, so the median is the second layer.
#[test]
#[ignore = "integration: GPU"]
fn median_depth_is_the_second_of_three_thirty_percent_layers() {
    let (device, queue) = &gpu();
    let renderer = Renderer::new(device).unwrap();
    let camera = camera(UVec2::new(33, 33));
    let depths = [2.0f32, 3.0, 4.0];
    let hexels = Hexels {
        variant: HexelVariant::V1,
        params: depths
            .iter()
            .flat_map(|&z| {
                hexel(
                    [0.0, 0.0, z],
                    Quat::IDENTITY,
                    [2.0, 2.0],
                    0.3,
                    200.0,
                    [[1.0; 3], [0.0; 3], [0.0; 3]],
                )
            })
            .collect(),
    };
    let logit = (0.3f32 / 0.7).ln();
    let splats = crate::Splats {
        transforms: depths
            .iter()
            .map(|&z| [0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .collect(),
        raw_opacities: vec![logit; 3],
        sh_coefficients: vec![[0.0; 3]; 3],
        sh_degree: 0,
        min_scale: None,
    };
    for scene in [
        renderer.upload_hexels(&hexels).unwrap(),
        renderer.upload(&splats).unwrap(),
    ] {
        let mut view = renderer.create_view(&scene, 1).unwrap();
        let pixels = 33 * 33;
        let (color, depth, normal) = (
            upload(device, &vec![[0.0f32; 4]; pixels]),
            upload(device, &vec![[0.0f32; 2]; pixels]),
            upload(device, &vec![[0.0f32; 4]; pixels]),
        );
        let target = Target::Surface {
            color: color.clone(),
            depth: depth.clone(),
            normal: normal.clone(),
        };
        let options = RenderOptions::default();
        render_fitted(
            device, queue, &renderer, &mut view, &camera, &options, &target,
        );
        let centre = 16 * 33 + 16;
        let alpha = read::<[f32; 4]>(device, queue, &color, pixels)[centre][3];
        let [expected, median] = read::<[f32; 2]>(device, queue, &depth, pixels)[centre];
        assert!((alpha - (1.0 - 0.343)).abs() < 1e-4, "alpha {alpha}");
        assert!((median - 3.0).abs() < 1e-5, "median {median}");
        let weighted = (2.0 * 0.3 + 3.0 * 0.21 + 4.0 * 0.147) / 0.657;
        assert!(
            (expected - weighted).abs() < 1e-4,
            "expected {expected} vs {weighted}"
        );
    }
}

/// Surface normals are world space and face the camera (upstream `hexels3d_forward.cu`: `n = t_u x t_v`,
/// negated when `dot(n, centre - eye) > 0`): a hexel facing away and one facing the camera give the same
/// normal, and a rotated camera reports it in world axes, not camera axes.
#[test]
#[ignore = "integration: GPU"]
fn hexel_normals_are_world_space_and_face_the_camera() {
    let size = UVec2::new(33, 33);
    let tilt = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.3);
    let camera = Camera {
        rotation: tilt,
        position: Vec3::new(0.3, -0.2, 0.1),
        ..camera(size)
    };
    let centre = camera.position + tilt * Vec3::new(0.0, 0.0, 2.0);
    let towards_camera = tilt * Vec3::NEG_Z;
    for facing in [Quat::IDENTITY, Quat::from_rotation_x(std::f32::consts::PI)] {
        let hexels = Hexels {
            variant: HexelVariant::V1,
            params: hexel(
                centre.to_array(),
                tilt * facing,
                [2.0, 2.0],
                0.9,
                200.0,
                [[1.0; 3], [0.0; 3], [0.0; 3]],
            ),
        };
        let normal = render(&hexels, &camera, &RenderOptions::default()).normal[16 * 33 + 16];
        assert!(
            (Vec3::from_slice(&normal[..3]) - towards_camera).length() < 1e-4,
            "normal {normal:?} vs {towards_camera:?}"
        );
        assert_eq!(normal[3], 0.0);
    }
}

/// Upstream 2D tile range (`gsplat_hexels` radius, gsplat `get_tile_bbox`), in f32 like the GPU.
fn image_tiles(h: &[f32], tiles: UVec2) -> [u32; 4] {
    let radius = (1.154_700_5 * h[2].abs().max(h[3].abs()) * (1.0 + 4.5 / h[5].max(1e-6)))
        .ceil()
        .max(1.0);
    let (center, r) = (Vec2::new(h[0], h[1]) / 16.0, radius / 16.0);
    let clamp = |v: f32, max: u32| (v as i32).max(0).unsigned_abs().min(max);
    [
        clamp(center.x - r, tiles.x),
        clamp(center.y - r, tiles.y),
        clamp(center.x + r + 1.0, tiles.x),
        clamp(center.y + r + 1.0, tiles.y),
    ]
}

/// Upstream top-10 normalized blend (`hexels_forward.cu`, colour-gradient kernel) in f64.
fn image_reference(hexels: &Hexels, size: UVec2) -> Vec<[f64; 4]> {
    let tiles = UVec2::new(size.x.div_ceil(16), size.y.div_ceil(16));
    let rows: Vec<&[f32]> = hexels
        .params
        .as_chunks::<15>()
        .0
        .iter()
        .map(|r| r.as_slice())
        .collect();
    let rects: Vec<[u32; 4]> = rows.iter().map(|h| image_tiles(h, tiles)).collect();
    let mut image = Vec::new();
    for y in 0..size.y {
        for x in 0..size.x {
            let (mut slots, mut min_slot, mut min_weight) = (Vec::<(usize, f64)>::new(), 0, -1.0);
            for (id, h) in rows.iter().enumerate() {
                let [x0, y0, x1, y1] = rects[id];
                if !(x0..x1).contains(&(x / 16)) || !(y0..y1).contains(&(y / 16)) {
                    continue;
                }
                let h: Vec<f64> = h.iter().map(|v| f64::from(*v)).collect();
                let (dx, dy) = (h[0] - f64::from(x), h[1] - f64::from(y));
                let (sin, cos) = h[4].sin_cos();
                let u = (dx * cos + dy * sin) / (h[2] + 1e-8);
                let v = (-dx * sin + dy * cos) / (h[3] + 1e-8);
                let k = 3f64.sqrt() * 0.5;
                let sdf = v
                    .abs()
                    .max((k * u + 0.5 * v).abs())
                    .max((k * u - 0.5 * v).abs())
                    - 1.0;
                let w = 1.0 / (1.0 + (sdf * h[5]).exp());
                if w > min_weight {
                    if slots.len() < 10 {
                        slots.push((id, w));
                        if slots.len() < 10 {
                            continue;
                        }
                    } else {
                        slots[min_slot] = (id, w);
                    }
                    (min_slot, min_weight) =
                        slots
                            .iter()
                            .enumerate()
                            .fold((0, f64::INFINITY), |best, (k, s)| {
                                if s.1 < best.1 { (k, s.1) } else { best }
                            });
                }
            }
            let total: f64 = slots.iter().map(|s| s.1).sum();
            let mut color = DVec3::ZERO;
            for (id, w) in &slots {
                let h: Vec<f64> = rows[*id].iter().map(|v| f64::from(*v)).collect();
                let (dx, dy) = (h[0] - f64::from(x), h[1] - f64::from(y));
                let c = DVec3::new(h[6], h[7], h[8])
                    + 0.1
                        * (dx * DVec3::new(h[9], h[10], h[11])
                            + dy * DVec3::new(h[12], h[13], h[14]));
                color += c * (w / (total + 1e-4));
            }
            image.push([color.x, color.y, color.z, total / (total + 1e-4)]);
        }
    }
    image
}

fn render_image(hexels: &Hexels, size: UVec2) -> Vec<[f32; 4]> {
    let (device, queue) = &gpu();
    let renderer = Renderer::new(device).unwrap();
    let scene = renderer.upload_hexels(hexels).unwrap();
    let mut view = renderer.create_view(&scene, 1).unwrap();
    let pixels = (size.x * size.y) as usize;
    let float = upload(device, &vec![[0.0f32; 4]; pixels]);
    let (camera, options) = (camera(size), RenderOptions::default());
    let target = Target::Float(float.clone());
    render_fitted(
        device, queue, &renderer, &mut view, &camera, &options, &target,
    );
    read(device, queue, &float, pixels)
}

#[test]
#[ignore = "integration: GPU"]
fn image_hexels_match_the_upstream_top_ten_blend() {
    let size = UVec2::new(96, 80);
    let mut rng = Lcg(11);
    let mut params = Vec::new();
    for _ in 0..400 {
        let row = [
            48.0 + 52.0 * rng.next(),
            40.0 + 44.0 * rng.next(),
            7.0 + 5.0 * rng.next(),
            7.0 + 5.0 * rng.next(),
            3.0 * rng.next(),
            9.5 + 3.5 * rng.next(),
        ];
        params.extend(row);
        params.extend((0..3).map(|_| 0.5 + 0.5 * rng.next()));
        params.extend((0..6).map(|_| 0.3 * rng.next()));
    }
    let hexels = Hexels {
        variant: HexelVariant::Image,
        params,
    };
    let gpu = render_image(&hexels, size);
    let cpu = image_reference(&hexels, size);
    let psnr = psnr(&gpu, &cpu);
    println!("2D GPU vs CPU {psnr:.1} dB");
    assert!(psnr >= 60.0, "2D GPU vs CPU PSNR {psnr:.2} dB");
}

#[test]
#[ignore = "integration: GPU"]
fn image_hexel_ties_keep_the_first_ten_in_primitive_order() {
    // Twelve identical footprints; strict replacement keeps primitives 0-9.
    let size = UVec2::new(32, 32);
    let params: Vec<f32> = (0..12)
        .flat_map(|i| {
            let mut row = vec![16.0, 16.0, 6.0, 6.0, 0.0, 10.0, i as f32, 0.0, 0.0];
            row.extend([0.0; 6]);
            row
        })
        .collect();
    let gpu = render_image(
        &Hexels {
            variant: HexelVariant::Image,
            params,
        },
        size,
    );
    let center = gpu[16 * 32 + 16];
    // Mean of 0..=9 = 4.5, scaled by sum / (sum + 1e-4).
    assert!((center[0] / center[3] - 4.5).abs() < 1e-4, "{center:?}");
}
