# Contract: hexel PLY (portable trained model)

Writer: `hexels_bench.hexel_ply` (Python, reads upstream checkpoints `ckpt_*.pt` with upstream code) and
`hexels_bench.image2d` for 2D fits. Readers: `gsplat-cli` (`packages/gsplat-rust-renderer/crates/gsplat-cli/src/hexel_ply.rs`,
Rust, no torch) and `gsplat_rust_renderer.hexels3d`. A change to the format edits this file together with the writer
and the readers.

## Format

Binary little-endian PLY, one `element vertex <N>` with one row per Hexel. All properties `float`
(f32). Values are ACTIVATED and render-ready (no logit/log), exactly the values the upstream
inference rasterizer consumes after its camera-independent precompute. The writer must use upstream
functions for the activations so that there is one source of truth.

Header comments (required):

```
comment hexels_format 1
comment hexels_variant v1|v2|v3
comment hexels_source <checkpoint path or run id>
comment hexels_step <training step>
```

Properties, in this order:

| Property | Meaning |
|---|---|
| `x y z` | center p (world) |
| `qw qx qy qz` | unit quaternion; tangent t_u = R e_x, t_v = R e_y, normal n = t_u x t_v |
| `su sv` | tangent scales (world units) |
| `opacity` | o in [0,1] |
| `sharpness` | beta > 0 (edge sharpness of the soft hexagon support K = sigmoid(-beta d)) |
| `c0_r c0_g c0_b` | base color c0 |
| `gu_r gu_g gu_b` | color gradient along u |
| `gv_r gv_g gv_b` | color gradient along v |
| `a_00 .. a_22` (9, row-major) | V2 only: linear view-dependent matrix A, residual A·d |
| `sh_0 .. sh_44` (45) | V3 only: SH degree 1..3 coefficients; order = upstream order, record it below |

Color: c(u,v) = c0 + u·g_u + v·g_v (+ A·d for V2, + SH(d) for V3). If upstream applies any
nonlinearity, clamp, offset (e.g. SH C0 = 0.28209 scaling, +0.5) or a convention for the view
direction d (camera-to-point or point-to-camera, normalized), the writer records it in the
"Upstream conventions" section below and the reader implements exactly that.

## Upstream conventions

- Hexagon SDF: d(u,v) = max(|v|, |sqrt(3)/2 u + v/2|, |sqrt(3)/2 u - v/2|) - 1 (paper). (u,v) are in
  units of (su, sv).
- Alpha: min(o·K, 0.99); skip if < 1/255; stop when transmittance < 1e-4; front-to-back over background.
- Tile bound: enclosing circle radius (2/sqrt(3))(1 + log(254)/beta) in (u,v) units (paper).
- Sort: per-tile by center depth (paper).

Verified 2026-10-28 against hexels@698d1af (`hexels/`), hexel-kernel@3af3220
(`gsplat/cuda_hexels3d/csrc/hexels3d_forward.cu`, below `fwd.cu`) and all 102 published fixed-sfm
`config.json` files. Paths are relative to the respective repo root.

- Stored vs. activated (`hexels/model.py`): checkpoint `model` state holds `means3d`, `log_scales`,
  `rotation_quats`, `colors`, `color_grads (P,2,3)`, `view_grads (P,K,3)`, `opacity_logits`,
  `log_softness`, `active_mask`. The writer emits only rows with `active_mask` true and applies exactly
  the module properties: `su sv = exp(log_scales)` (L.184), `q = rotation_quats / |rotation_quats|`
  (L.188, `F.normalize`, eps 1e-12), `opacity = sigmoid(opacity_logits)` (L.180),
  `sharpness = exp(log_softness)` (L.193). Colors and gradients are raw (no activation).
  Checkpoints with children (`child_parent_idx` non-empty), `softness_mode = "aniso"`, a compact
  `deployment` payload, `scale_modifier != 1`, `prefilter_edge_on` or a `color_model` other than
  `linear` are rejected by the writer (none of the 102 published fixed-sfm models has any of them).
- Variant from `config.json` (same rule as upstream `infer_appearance_layout`,
  `hexels/compact_checkpoint.py` L.59): `view_model "none"` = V1; `"linear"` + `use_view_grads` = V2;
  `"sh3"` + `use_view_grads` = V3.
- Quaternion: `(qw, qx, qy, qz)`, w first (`fwd.cu` L.101-127, the 2DGS packing with w in lane x).
  R is the standard unit-quaternion rotation; `t_u = R[:,0]`, `t_v = R[:,1]`, `n = R[:,2]` (L.548).
  The splat-to-world map is `L = R · diag(su, sv, ·)` plus the center (L.156-202).
- Color field, per pixel inside the kernel (`fwd.cu` L.876-895): `c = c0 + u·g_u + v·g_v`, with (u,v)
  the ray-plane intersection in (su, sv) units (the same (u,v) the SDF uses). No activation, no clamp,
  no offset on c or on the composited pixel; the float image goes out unclamped (the evaluator clamps
  to [0,1] once, `hexels/benchmark_metrics.py` L.359). Unlike 3DGS/Brush there is no `max(c, 0)`.
  Layout: `g_u = color_grads[:, 0, :]`, `g_v = color_grads[:, 1, :]`.
- View direction d (V2 and V3 only): ONE direction per hexel, at its center, camera to point:
  `d = (x - campos) / sqrt(|x - campos|^2 + 1e-8)` (`hexels/renderer.py` L.175-179). The view term is
  added to c0 once per hexel per camera (renderer.py L.180-182) before the kernel runs, so it is
  constant across the hexel; the kernel then sees `view_model = "none"` (renderer.py L.104).
- V2: `c0 += A·d`, `A[i][j] = view_grads[p, i, j]` (i = RGB channel, j = x,y,z of d;
  `torch.bmm(view_grads, d)`, renderer.py L.18-19). PLY `a_ij` row-major = `A[i][j]`.
- V3: `c0 += sum_k basis_k(d) · coeffs[k][ch]`, k = 0..14, degree 1..3 only (no DC term, no C0 =
  0.28209, no +0.5), all 15 used at render (`sh_degree = 3`). Basis order (renderer.py L.56-82), with
  C1 = 0.4886025119029199, C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
  -1.0925484305920792, 0.5462742152960396), C3 = (-0.5900435899266435, 2.890611442640554,
  -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435):
  `[-C1 y, C1 z, -C1 x, C2[0] xy, C2[1] yz, C2[2](2zz-xx-yy), C2[3] xz, C2[4](xx-yy),
  C3[0] y(3xx-yy), C3[1] xyz, C3[2] y(4zz-xx-yy), C3[3] z(2zz-3xx-3yy), C3[4] x(4zz-xx-yy),
  C3[5] z(xx-yy), C3[6] x(xx-3yy)]` (the 3DGS order). PLY `sh_(3k+ch) = view_grads[p, k, ch]`
  (coefficient-major, RGB innermost).
- Sharpness: the kernel clamps `beta = max(beta, 1e-4)` (`fwd.cu` L.444) before use; the PLY stores the
  unclamped `exp(log_softness)`.
- Support and alpha (`fwd.cu` L.255-281, L.801-845): `K = 1 / (1 + exp(d·beta))`,
  `alpha = min(o·K, 0.99)`, skip if `alpha < 1/255`, skip the ray-plane hit if its view depth < 0.2
  (near plane, also the center cull, L.23/L.98), stop before adding a hexel when
  `T·(1 - alpha) < 1e-4`. Pixel = `C + T·background`.
- Pixel coordinates (`fwd.cu` L.196): the kernel evaluates pixel i at coordinate i with principal point
  (W-1)/2, i.e. the same rays as the pixel-center-at-i+0.5 convention with cx = W/2. Upstream ignores
  COLMAP's cx/cy (2DGS `getProjectionMatrix`); znear 0.01, zfar 100 in the projection matrix.
- Tile coverage (`conservative_bounds = false` in every published config): the intersection of the legacy
  2DGS AABB (circle of radius `(2/sqrt3)(1 + log254/beta)` through `compute_aabb`, L.449-467) and the
  projected-hexagon bound with apothem `1 + log(255(1+8eps)o - 1)/beta` (L.470-490). Upstream keeps
  the intersection so old checkpoints render identically; a WGSL bound that is larger renders extra
  pixels where upstream did not.
- Depth sort key: view-space z of the center (`fwd.cu` L.497, L.578).
- Background per dataset: black `(0, 0, 0)` for all published NVS models (DTU, Mip-NeRF 360, T&T:
  `white_background = false`, `random_background = false` in all 102 fixed-sfm configs;
  `hexels/render.py` L.388). The writer puts it in `cameras.json` (`background`).
- Precompute summary: everything camera-independent is in the PLY already activated; the only
  per-camera precompute is the V2/V3 per-hexel view term above.

## 2D addendum: Image-GS-style fits (`hexels_variant 2d`)

Writer: `hexels_bench.image2d`. 2D fits are not alpha-composited 3D hexels: upstream decodes them with
`rasterize_hexels_sum` (hexel-kernel `gsplat/cuda/csrc/hexels_forward.cu`, top-K = 10 normalized blend), so a 2D
PLY is its own variant, not a z = 0 plane.

Header comments: `hexels_format 1`, `hexels_variant 2d`, `hexels_source`, `hexels_step`,
`hexels_image <width> <height>`, `hexels_color_grad_scale 0.1`. One float row per hexel:

| Property | Meaning (upstream `scripts/image_compression_stable.py` L.111-150) |
|---|---|
| `x y` | center in pixels = `positions * (W, H)` (the kernel's `xys`; pixel i is evaluated at coordinate i) |
| `su sv` | `exp(log_scales)`, pixels |
| `theta` | rotation, radians (kernel: `u = dx cos + dy sin`, `v = -dx sin + dy cos`, then `/ (su, sv)`, `hexels_forward.cu` L.58-69) |
| `sharpness` | softness, already softplus'ed in the file; 10.0 for flat (`flat8`) fits |
| `c0_r c0_g c0_b` | `features` |
| `gx_* gy_*` | raw `color_gradients[:, 0]`, `[:, 1]` (zero for flat fits) |

Color at pixel p (`hexels_forward.cu` L.490-495): `c = c0 + 0.1 * ((x - p_x) * gx + (y - p_y) * gy)`:
the offset is center minus pixel, in image pixels and image axes (not the hexel's u/v), scaled by
`hexels_color_grad_scale`. The blend weights and the top-K rule follow.

### 2D renderer rules

The renderer reads exactly the schema above: `hexels_variant 2d`, `hexels_image <W> <H>`, 15 floats
`x y su sv theta sharpness c0_* gx_* gy_*` (zero gradients for flat fits), rows in upstream primitive order.
WGSL rules (upstream `hexels_forward.cu` L.22-95, L.340-517; `gsplat_hexels.py` L.55-137):
pixel p = (j, i) integer, d = m - p; u = (d.x cos + d.y sin)/(su + 1e-8), v = (-d.x sin + d.y cos)/(sv + 1e-8);
w = 1/(1 + exp(sdf(u, v) beta)); c = c0 + 0.1 (d.x gx + d.y gy); keep the 10 largest w per pixel (fill empty
slots first, then replace the minimum on strict >), candidates in ascending primitive index;
C = sum w c / (sum w + 1e-4); radius r = max(ceil((2/sqrt3) max(su, sv)(1 + 4.5/beta)), 1);
tiles x0 = clamp(int(x/16 - r/16), 0, tiles_x), x1 = clamp(int(x/16 + r/16 + 1), 0, tiles_x) (same for y).
