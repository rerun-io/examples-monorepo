//! Tile coverage against brute force over pixel centres.
use glam::{BVec2, DMat2, DVec2, UVec2, UVec4, Vec2, Vec3};

const TILE: f32 = 16.0;

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

#[test]
fn rejected_tiles_have_no_pixel_centre_inside_the_splat() {
    let tiles = UVec2::new(6, 5);
    let mut rng = Lcg(7);
    let mut tile_hits = 0;
    for _ in 0..4000 {
        let (p, c, power) = random_splat(&mut rng, (-24.0, 120.0));
        let bb = tile_bbox(p, bbox_extent(c, power), tiles);
        for ty in 0..tiles.y {
            for tx in 0..tiles.x {
                let origin = UVec2::new(tx, ty) * 16;
                let centres = |size: UVec2, at: UVec2| {
                    (0..size.y).flat_map(move |y| {
                        (0..size.x).map(move |x| (at + UVec2::new(x, y)).as_vec2() + 0.5)
                    })
                };
                let inside_bbox = (bb.x..bb.z).contains(&tx) && (bb.y..bb.w).contains(&ty);
                if !inside_bbox {
                    assert_rejection_is_safe(
                        centres(UVec2::splat(16), origin),
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
                        centres(UVec2::splat(16), origin),
                        p,
                        c,
                        power,
                        "tile",
                    );
                }
            }
        }
    }
    // Most tiles miss; enough must hit that both outcomes are exercised.
    assert!(tile_hits > 4000, "{tile_hits} tiles hit");
}
