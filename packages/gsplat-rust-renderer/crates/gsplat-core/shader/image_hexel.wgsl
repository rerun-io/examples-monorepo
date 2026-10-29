#import <./common.wgsl>
#import <./hexagon.wgsl>

// 2D image hexels (Image-GS style), hand-ported from hexel-kernel 3af3220
// gsplat/cuda/csrc/hexels_forward.cu (nd_rasterize_forward_topk_norm_hexels_color_grad)
// and the tile radius of honeycomb-surfels/hexels gsplat_hexels.py. Pixel centres are
// integer coordinates. Parameters per hexel: x y su sv theta sharpness, colour, x/y gradients.
const TOP_K = 10u;

// The 2D hexel is the `Primitive` of the shared tile mapper; 128 of 64 bytes per raster batch.
alias Primitive = ImageHexel;
const BATCH = 128u;

struct ImageHexel {
    x: f32,
    y: f32,
    su: f32,
    sv: f32,
    cos_theta: f32,
    sin_theta: f32,
    sharpness: f32,
    color: array<f32, 9>,
}

// Circumscribed radius at the 3-sigma-like margin 4.5 / beta, then gsplat's get_tile_bbox.
fn image_tiles(x: f32, y: f32, su: f32, sv: f32, sharpness: f32) -> vec4u {
    let radius = max(ceil(HEX_RADIUS * max(abs(su), abs(sv)) * (1.0 + 4.5 / max(sharpness, 1e-6))), 1.0);
    let grid = vec2i(i32(frame.image.tiles_x), i32(frame.image.tiles_y));
    let center = vec2f(x, y) / f32(TILE_WIDTH);
    let r = radius / f32(TILE_WIDTH);
    return vec4u(vec2u(clamp(vec2i(center - r), vec2i(0), grid)), vec2u(clamp(vec2i(center + r + 1.0), vec2i(0), grid)));
}

fn primitive_tiles(h: ImageHexel) -> vec4u {
    return image_tiles(h.x, h.y, h.su, h.sv, h.sharpness);
}
