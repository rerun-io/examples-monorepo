#import <./common.wgsl>
#import <./hexagon.wgsl>

// Hexel forward model hand-ported from honeycomb-surfels/hexel-kernel 3af3220,
// gsplat/cuda_hexels3d/csrc/hexels3d_forward.cu (preprocess_hexels3d, render_hexels3d).
// Pixel centres are integer coordinates, as upstream; the principal point moves by -0.5.
const HEX_NEAR = 0.2;
const LOG_254 = 5.537334267;
const FILTER_SIZE = 0.707106;
const FLT_EPSILON = 1.1920929e-7;

// Rows of the plane-to-pixel map: (u, v, 1) . tu = x w, (u, v, 1) . tv = y w, (u, v, 1) . tw = w.
// `color` holds the base colour (with the view term folded in), then the u and v gradients.
struct Hexel {
    t: array<f32, 9>,
    opacity: f32,
    sharpness: f32,
    color: array<f32, 9>,
}

struct Plane { tu: vec3f, tv: vec3f, tw: vec3f, }

// The hexel is the `Primitive` of the shared tile mapper and rasterizer.
alias Primitive = Hexel;
const BATCH = 128u; // 128 hexels of 80 bytes fill the core's 10 KiB workgroup budget.

fn primitive_tiles(h: Hexel) -> vec4u {
    return hexel_tiles(plane(h), h.opacity, h.sharpness);
}

fn plane(h: Hexel) -> Plane {
    return Plane(vec3f(h.t[0], h.t[1], h.t[2]), vec3f(h.t[3], h.t[4], h.t[5]), vec3f(h.t[6], h.t[7], h.t[8]));
}

// Upstream get_rect: tile range of a pixel box, truncating toward zero.
fn tile_rect(p: vec2f, radius: vec2f) -> vec4u {
    let grid = vec2i(i32(frame.image.tiles_x), i32(frame.image.tiles_y));
    let lo = min(grid, max(vec2i(0), vec2i((p - radius) / f32(TILE_WIDTH))));
    let hi = min(grid, max(vec2i(0), vec2i((p + radius + f32(TILE_WIDTH) - 1.0) / f32(TILE_WIDTH))));
    return vec4u(vec2u(lo), vec2u(hi));
}

// Upstream compute_hex_bounds: hexagon of the given apothem clipped at the near plane.
fn hexagon_tiles(m: Plane, apothem_in: f32) -> vec4u {
    let apothem = apothem_in + 4e-5 * max(1.0, abs(apothem_in));
    if apothem <= 0.0 {
        return vec4u(0u);
    }
    let r = HEX_RADIUS * apothem;
    var corners = array<vec2f, 6>(vec2f(r, 0.0), vec2f(0.5 * r, apothem), vec2f(-0.5 * r, apothem),
        vec2f(-r, 0.0), vec2f(-0.5 * r, -apothem), vec2f(0.5 * r, -apothem));
    var lower = vec2f(3.4028235e38);
    var upper = vec2f(-3.4028235e38);
    for (var i = 0u; i < 6u; i++) {
        let a = vec3f(corners[i], 1.0);
        let b = vec3f(corners[(i + 1u) % 6u], 1.0);
        let za = dot(a, m.tw);
        let zb = dot(b, m.tw);
        if za >= HEX_NEAR {
            let p = vec2f(dot(a, m.tu) / za, dot(a, m.tv) / za);
            lower = min(lower, p);
            upper = max(upper, p);
        }
        if (za < HEX_NEAR) != (zb < HEX_NEAR) {
            let e = a + ((HEX_NEAR - za) / (zb - za)) * (b - a);
            let p = vec2f(dot(e, m.tu) / HEX_NEAR, dot(e, m.tv) / HEX_NEAR);
            lower = min(lower, p);
            upper = max(upper, p);
        }
    }
    let size = vec2f(f32(frame.image.width), f32(frame.image.height));
    if lower.x > size.x || lower.y > size.y || upper.x < -1.0 || upper.y < -1.0 {
        return vec4u(0u);
    }
    lower = max(lower - 1.0, vec2f(-f32(TILE_WIDTH)));
    upper = min(upper + 1.0, size + f32(TILE_WIDTH));
    return tile_rect((lower + upper) * 0.5, ceil((upper - lower) * 0.5) + 1.0);
}

// Upstream default coverage (conservative_bounds = false): the legacy 2DGS box of the
// enclosing circle, intersected with the hexagon where alpha can reach 1/255.
// An empty rectangle has min == max.
fn hexel_tiles(m: Plane, opacity: f32, sharpness: f32) -> vec4u {
    let cutoff = HEX_RADIUS * (1.0 + LOG_254 / sharpness);
    let t = vec3f(cutoff * cutoff, cutoff * cutoff, -1.0);
    let d = dot(t, m.tw * m.tw);
    if d == 0.0 || opacity < ALPHA_CUTOFF {
        return vec4u(0u);
    }
    let f = (1.0 / d) * t;
    let center = vec2f(dot(f, m.tu * m.tw), dot(f, m.tv * m.tw));
    let extent = sqrt(max(vec2f(1e-4), center * center - vec2f(dot(f, m.tu * m.tu), dot(f, m.tv * m.tv))));
    // (int)ceilf saturates on CUDA; clamp before the float round trip.
    let radius = min(ceil(max(max(extent.x, extent.y), cutoff * FILTER_SIZE)), 2147483648.0);
    var rect = tile_rect(center, vec2f(radius));
    if !(rect.x < rect.z && rect.y < rect.w) {
        return vec4u(0u);
    }
    if d < 0.0 && opacity <= 1.0 {
        rect = tile_rect(center, min(vec2f(radius), ceil(max(extent, vec2f(cutoff * FILTER_SIZE))) + 1.0));
    }
    var margin = LOG_254 / sharpness;
    if opacity > ALPHA_CUTOFF {
        margin = log((255.0 * (1.0 + 8.0 * FLT_EPSILON)) * opacity - 1.0) / sharpness;
    }
    let hexagon = hexagon_tiles(m, 1.0 + margin);
    rect = vec4u(max(rect.xy, hexagon.xy), min(rect.zw, hexagon.zw));
    if !(rect.x < rect.z && rect.y < rect.w) {
        return vec4u(0u);
    }
    return rect;
}
