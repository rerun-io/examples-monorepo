//! Tile and block coverage against brute force over pixel centres, and the raster's block masks against the same
//! raster with every block reached.
use crate::kernels::TILE_WIDTH;
use crate::test_utils::{gpu, pinhole_camera, read, render_fitted, upload};
use crate::{RenderOptions, Renderer, Splats, Target};
use glam::{BVec2, DMat2, DVec2, Quat, UVec2, UVec4, Vec2, Vec3};

const TILE: f32 = TILE_WIDTH as f32;

// Line for line the WGSL of common.wgsl (`sigma`, `bbox_extent`, `tile_bbox`, `rect_hit`), so the brute-force
// checks below hold the shader expressions; edit both together.
#[expect(clippy::manual_midpoint, reason = "the WGSL expression")]
fn sigma(p: Vec2, center: Vec2, conic: Vec3) -> f32 {
    let d = p - center;
    0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y
}

fn bbox_extent(c: Vec3, power: f32) -> Vec2 {
    let det = c.x * c.z - c.y * c.y;
    if det <= 0.0 {
        return Vec2::splat(-1.0);
    }
    (2.0 * power * Vec2::new(c.z, c.x) * (1.0 / det)).map(f32::sqrt)
}

fn tile_bbox(p: Vec2, extent: Vec2, tiles: UVec2) -> UVec4 {
    let bounds = tiles.as_vec2();
    let lo = ((p - extent - (TILE - 0.5)) / TILE).ceil();
    let hi = ((p + extent - 0.5) / TILE).floor() + 1.0;
    let (lo, hi) = (
        lo.clamp(Vec2::ZERO, bounds).as_uvec2(),
        hi.clamp(Vec2::ZERO, bounds).as_uvec2(),
    );
    UVec4::new(lo.x, lo.y, hi.x, hi.y)
}

#[expect(clippy::nonminimal_bool, reason = "the WGSL expression")]
fn rect_hit(lo: Vec2, hi: Vec2, p: Vec2, c: Vec3, power: f32) -> bool {
    let left = p.x < lo.x;
    let above = p.y < lo.y;
    if !(left || p.x > hi.x) && !(above || p.y > hi.y) {
        return true;
    }
    let corner = Vec2::new(
        if left { lo.x } else { hi.x },
        if above { lo.y } else { hi.y },
    );
    let d = Vec2::select(BVec2::new(left, above), hi - lo, lo - hi);
    let diff = p - corner;
    let tx = ((c.x * diff.x + c.y * diff.y) / (c.x * d.x)).clamp(0.0, 1.0);
    let ty = ((c.y * diff.x + c.z * diff.y) / (c.z * d.y)).clamp(0.0, 1.0);
    sigma(corner + Vec2::new(tx * d.x, 0.0), p, c).min(sigma(
        corner + Vec2::new(0.0, ty * d.y),
        p,
        c,
    )) <= power
}

struct Lcg(u64);
impl Lcg {
    /// Uniform in [0, 1).
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

/// A random eccentric screen-space Gaussian: centre, conic and power as `map_tiles` computes them.
fn random_splat(rng: &mut Lcg, span: (f64, f64)) -> (Vec2, Vec3, f32) {
    let center = Vec2::new(
        rng.range(span.0, span.1) as f32,
        rng.range(span.0, span.1) as f32,
    );
    let major = rng.range(0.3, 12.0);
    let minor = major * rng.range(0.03, 1.0);
    let (sin, cos) = rng.range(0.0, std::f64::consts::PI).sin_cos();
    let rotation = DMat2::from_cols(DVec2::new(cos, sin), DVec2::new(-sin, cos));
    let covariance = rotation
        * DMat2::from_diagonal(DVec2::new(major * major, minor * minor))
        * rotation.transpose();
    let conic = covariance.inverse();
    let opacity = rng.range(1.02 / 255.0, 1.0) as f32;
    (
        center,
        Vec3::new(
            conic.x_axis.x as f32,
            conic.y_axis.x as f32,
            conic.y_axis.y as f32,
        ),
        (opacity * 255.0).ln(),
    )
}

/// Every centre misses: its sigma lies above power, beyond a 1-ulp band (`rect_hit`'s closest point and a pixel
/// centre can round apart).
fn assert_rejection_is_safe(
    centres: impl Iterator<Item = Vec2>,
    p: Vec2,
    c: Vec3,
    power: f32,
    what: &str,
) {
    let band = power.abs().max(f32::MIN_POSITIVE) * f32::EPSILON;
    for centre in centres {
        let s = sigma(centre, p, c);
        assert!(
            s > power - band,
            "{what} rejected, yet centre {centre} has sigma {s} <= power {power} (splat at {p}, conic {c})"
        );
    }
}

fn block_origin(b: u32) -> UVec2 {
    // pixel(tile, 32b) in raster_outputs.wgsl: x takes lid bits 0, 2, 4, 6 and y bits 1, 3, 5, 7.
    UVec2::new(8 * ((b >> 1) & 1), 4 * (b & 1) + 8 * ((b >> 2) & 1))
}

#[test]
fn rejected_tiles_and_blocks_have_no_pixel_centre_inside_the_splat() {
    let tiles = UVec2::new(6, 5);
    let mut rng = Lcg(7);
    let (mut tile_hits, mut block_hits) = (0, 0);
    for _ in 0..4000 {
        let (p, c, power) = random_splat(&mut rng, (-24.0, 120.0));
        let bb = tile_bbox(p, bbox_extent(c, power), tiles);
        for ty in 0..tiles.y {
            for tx in 0..tiles.x {
                let origin = UVec2::new(tx, ty) * TILE_WIDTH;
                let centres = |size: UVec2, at: UVec2| {
                    (0..size.y).flat_map(move |y| {
                        (0..size.x).map(move |x| (at + UVec2::new(x, y)).as_vec2() + 0.5)
                    })
                };
                let inside_bbox = (bb.x..bb.z).contains(&tx) && (bb.y..bb.w).contains(&ty);
                if !inside_bbox {
                    assert_rejection_is_safe(
                        centres(UVec2::splat(TILE_WIDTH), origin),
                        p,
                        c,
                        power,
                        "tile outside tile_bbox",
                    );
                }
                let lo = origin.as_vec2() + 0.5;
                if rect_hit(lo, lo + (TILE - 1.0), p, c, power) {
                    tile_hits += 1;
                } else {
                    assert_rejection_is_safe(
                        centres(UVec2::splat(TILE_WIDTH), origin),
                        p,
                        c,
                        power,
                        "tile",
                    );
                }
                for b in 0..8 {
                    let at = origin + block_origin(b);
                    let lo = at.as_vec2() + 0.5;
                    if rect_hit(lo, lo + Vec2::new(7.0, 3.0), p, c, power) {
                        block_hits += 1;
                    } else {
                        assert_rejection_is_safe(
                            centres(UVec2::new(8, 4), at),
                            p,
                            c,
                            power,
                            "8x4 block",
                        );
                    }
                }
            }
        }
    }
    // Most rectangles miss; enough must hit that both outcomes are exercised.
    assert!(
        tile_hits > 4000 && block_hits > 4 * tile_hits,
        "{tile_hits} tiles, {block_hits} blocks hit"
    );
}

/// Small eccentric splats at 8x4 block and 16x16 tile borders, several batches deep in each tile.
fn border_splats(focal: Vec2, size: u32) -> Splats {
    let mut rng = Lcg(3);
    let mut splats = Splats::default();
    let center = size as f32 * 0.5;
    for _ in 0..6 {
        for v in (0..size).step_by(4) {
            for u in (0..size).step_by(8) {
                let depth = rng.range(2.0, 4.0) as f32;
                let jitter = Vec2::new(rng.range(-1.5, 1.5) as f32, rng.range(-1.5, 1.5) as f32);
                let xy = (UVec2::new(u, v).as_vec2() + jitter - center) * depth / focal;
                // A pixel is about depth / focal = 0.05 world units: about 0.3 to 4 pixels long, mostly thin.
                let pixel = depth / focal.x;
                let long = pixel * rng.range(0.3, 4.0) as f32;
                let thin = long * rng.range(0.05, 1.0) as f32;
                let q = Quat::from_xyzw(
                    rng.range(-1.0, 1.0) as f32,
                    rng.range(-1.0, 1.0) as f32,
                    rng.range(-1.0, 1.0) as f32,
                    rng.range(-1.0, 1.0) as f32,
                )
                .normalize();
                splats.transforms.push([
                    xy.x,
                    xy.y,
                    depth,
                    q.w,
                    q.x,
                    q.y,
                    q.z,
                    long.ln(),
                    thin.ln(),
                    thin.ln(),
                ]);
                splats.raw_opacities.push(rng.range(-3.0, 2.0) as f32);
                splats.sh_coefficients.push([
                    rng.range(-1.5, 1.5) as f32,
                    rng.range(-1.5, 1.5) as f32,
                    rng.range(-1.5, 1.5) as f32,
                ]);
            }
        }
    }
    splats
}

/// The border scene rendered by `renderer` to a float target, as bits.
fn render_border_splats(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer,
    camera: &crate::Camera,
) -> Vec<u32> {
    let splats = border_splats(camera.focal(), camera.size.x);
    let pixels = (camera.size.x * camera.size.y) as usize;
    let scene = renderer.upload(&splats).unwrap();
    let mut view = renderer.create_view(&scene, 1).unwrap();
    let float = upload(device, &vec![[0.0f32; 4]; pixels]);
    render_fitted(
        device,
        queue,
        renderer,
        &mut view,
        camera,
        &RenderOptions::default(),
        &Target::Float(float.clone()),
    );
    read::<[f32; 4]>(device, queue, &float, pixels)
        .into_iter()
        .flatten()
        .map(f32::to_bits)
        .collect()
}

fn covered_pixels(bits: &[u32]) -> usize {
    bits.chunks(4)
        .filter(|rgba| f32::from_bits(rgba[3]) > 0.5)
        .count()
}

fn differ(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(a, b)| a != b).count()
}

#[test]
#[ignore = "integration: GPU"]
fn block_masks_skip_only_splats_that_miss_the_block() {
    let (device, queue) = &gpu();
    // An image size that is not a multiple of 16 keeps partial edge tiles in the test.
    let camera = pinhole_camera(72);
    let pixels = (camera.size.x * camera.size.y) as usize;
    let render = |source: Option<String>| {
        let renderer = Renderer::new(device).unwrap();
        let renderer = match source {
            Some(source) => renderer.with_splat_raster(&source),
            None => renderer,
        };
        render_border_splats(device, queue, &renderer, &camera)
    };
    let source = crate::shader::resolve_variant(
        "raster.wgsl",
        device.features().contains(wgpu::Features::SUBGROUP),
    );
    let mask = "return (blocks[t] & (1u << (lid >> 5u))) != 0u;";
    assert_eq!(source.matches(mask).count(), 1);
    let masked = render(None);
    let every_block = render(Some(source.replace(mask, "return true;")));
    // The test must see a wrong block: with the block index off by one row of blocks, pixels lose contributors.
    let wrong_block = render(Some(source.replace(
        mask,
        "return (blocks[t] & (1u << ((lid >> 5u) ^ 1u))) != 0u;",
    )));
    let covered = covered_pixels(&every_block);
    assert!(
        covered > pixels / 2,
        "only {covered} of {pixels} pixels covered"
    );
    assert_eq!(differ(&masked, &every_block), 0, "the masks changed pixels");
    assert!(
        differ(&wrong_block, &every_block) > 0,
        "a wrong block index went unnoticed"
    );
}

/// The WGSL `tile_hit` itself (the CPU copy above checks only the Rust transcription): rejecting tiles must
/// not change a pixel against the same render with every tile of the bounding box hit.
#[test]
#[ignore = "integration: GPU"]
fn tile_hit_skips_only_tiles_that_no_pixel_centre_reaches() {
    let (device, queue) = &gpu();
    let camera = pinhole_camera(72);
    let pixels = (camera.size.x * camera.size.y) as usize;
    let subgroups = device.features().contains(wgpu::Features::SUBGROUP);
    let (project, map) = (
        crate::shader::resolve_variant("project.wgsl", subgroups),
        crate::shader::resolve_variant("map.wgsl", subgroups),
    );
    let test = "return rect_hit(lo, lo + f32(TILE_WIDTH - 1u), p, c, power);";
    assert_eq!(project.matches(test).count(), 1);
    assert_eq!(map.matches(test).count(), 1);
    // Projection counts and mapping emits with the same patched test, as the stock kernels do.
    let render = |replacement: Option<&str>| {
        let renderer = Renderer::new(device).unwrap();
        let renderer = match replacement {
            Some(r) => renderer.with_splat_tiles(&project.replace(test, r), &map.replace(test, r)),
            None => renderer,
        };
        render_border_splats(device, queue, &renderer, &camera)
    };
    let tested = render(None);
    let every_tile = render(Some("return true;"));
    // The test must see a wrong tile: tested one tile to the right, pixels lose contributors.
    let wrong_tile = render(Some(
        "return rect_hit(lo + vec2f(f32(TILE_WIDTH), 0.0), lo + vec2f(f32(2u * TILE_WIDTH - 1u), f32(TILE_WIDTH - 1u)), p, c, power);",
    ));
    let covered = covered_pixels(&every_tile);
    assert!(
        covered > pixels / 2,
        "only {covered} of {pixels} pixels covered"
    );
    assert_eq!(differ(&tested, &every_tile), 0, "tile_hit changed pixels");
    assert!(
        differ(&wrong_tile, &every_tile) > 0,
        "a wrong tile test went unnoticed"
    );
}
