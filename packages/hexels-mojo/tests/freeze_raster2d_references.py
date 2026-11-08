"""Freeze hexel-kernel's 2D hexel rasterizer (flat and color-gradient) outputs for hexels_mojo's raster2d tests.

Needs hexel-kernel (`gsplat`, still built by recipes/hexels-src) and `gsplat_hexels.py` for its radius and tile
helpers. The kernels are called on `gsplat.cuda` / `gsplat.utils` directly: `gsplat_hexels`'s own handles serve
hexels_mojo's ports since patch 0005. Rewrites the `/raster2d/`, `/binning/` and `/gaussians2d/` (Image-GS's Gaussian kernels) records of
`fixtures/hexels_references.rrd` (the other records are copied over unchanged).

    python tests/freeze_raster2d_references.py --root <hexels-src checkout>
"""

import argparse
import sys
from pathlib import Path

import rerun as rr
import torch
from hexels_references import REFERENCE_FILE, encode, load  # pyrefly: ignore[missing-import]
from torch import Tensor

# (name, color gradients, N, H, W, softness: "tensor" | "triplet" | float, sharpness, emission, aa_kernel_size, seed)
CASES = (
    ("rich", True, 1500, 128, 160, "tensor", False, False, 0.0, 10),
    ("triplet_peak_emission_aa", True, 800, 96, 112, "triplet", True, True, 0.3, 11),
    ("scalar", True, 600, 80, 96, 10.0, False, False, 0.0, 12),
    ("flat", False, 1200, 112, 128, "tensor", False, False, 0.0, 13),
    ("flat_scalar_peak_emission", False, 600, 80, 96, 10.0, True, True, 0.2, 14),
)


def _case(gh, utils, cuda, name, color_grad, n, h, w, softness_kind, with_sharpness, with_emission, aa, seed) -> dict[str, dict[str, Tensor]]:
    g = torch.Generator().manual_seed(seed)
    u = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    xys = (u(n, 2) * torch.tensor([w, h])).float().cuda()
    scales = (1.5 + 6.0 * u(n, 2)).float().cuda()
    rotations = ((u(n, 1) - 0.5) * 6.283).float().cuda()
    colors = u(n, 3).float().cuda()
    color_grads = ((u(n, 2, 3) - 0.5) * 0.2).float().cuda()
    if softness_kind == "tensor":
        softness: float | Tensor = (4.0 + 12.0 * u(n)).float().cuda()
    elif softness_kind == "triplet":
        softness = (4.0 + 12.0 * u(n, 3)).float().cuda()
    else:
        softness = float(softness_kind)
    sharpness = (0.5 * u(n)).float().cuda() if with_sharpness else None
    emission = (0.2 * u(n, 3)).float().cuda() if with_emission else None
    radii = gh._compute_hexel_radii(scales, softness)
    num_tiles_hit = gh._compute_num_tiles_hit(xys, radii, h, w, 16, 16)
    num_intersects, cum_tiles_hit = utils.compute_cumulative_intersects(num_tiles_hit)
    tile_bounds = ((w + 15) // 16, (h + 15) // 16, 1)
    isect_ids, gaussian_ids, isect_sorted, gaussian_ids_sorted, tile_bins = utils.bin_and_sort_gaussians(
        n, num_intersects, xys, radii, cum_tiles_hit, tile_bounds
    )
    tensor_softness = isinstance(softness, Tensor)
    empty = torch.empty(0, device="cuda", dtype=torch.float32)
    sharp_t = sharpness if sharpness is not None else empty
    emit_t = emission if emission is not None else empty
    if not color_grad:
        geo = (tile_bounds, (16, 16, 1), (w, h, 1), gaussian_ids_sorted, tile_bins, xys, scales, rotations, colors, aa)
        if tensor_softness:
            out_img, pixel_topk, weights_sum, entropy = cuda.nd_rasterize_forward_topk_norm_hexels_softness(*geo, softness, 0.0, sharp_t, emit_t)
        else:
            out_img, pixel_topk, weights_sum, entropy = cuda.nd_rasterize_forward_topk_norm_hexels(*geo, float(softness), sharp_t, emit_t)
    elif tensor_softness:
        out_img, pixel_topk, weights_sum, entropy = cuda.nd_rasterize_forward_topk_norm_hexels_color_grad_softness(
            tile_bounds, (16, 16, 1), (w, h, 1), gaussian_ids_sorted, tile_bins, xys, scales, rotations, colors, color_grads, 0.1, aa,
            softness, 0.0, sharp_t, emit_t,
        )  # fmt: skip
    else:
        out_img, pixel_topk, weights_sum, entropy = cuda.nd_rasterize_forward_topk_norm_hexels_color_grad(
            tile_bounds, (16, 16, 1), (w, h, 1), gaussian_ids_sorted, tile_bins, xys, scales, rotations, colors, color_grads, 0.1, aa,
            float(softness), sharp_t, emit_t,
        )  # fmt: skip
    v_out = torch.randn(h, w, 3, generator=g).float().cuda()
    v_ws = (0.1 * torch.randn(h, w, generator=g)).float().cuda()
    v_ent = (0.1 * torch.randn(h, w, generator=g)).float().cuda()
    if not color_grad:
        bgeo = (h, w, 16, 16, gaussian_ids_sorted, tile_bins, xys, scales, rotations, colors)
        if tensor_softness:
            grads = cuda.nd_rasterize_backward_topk_norm_hexels_softness(
                *bgeo, softness, v_out, v_ws, v_ent, pixel_topk.contiguous(), aa, 0.0, sharp_t, emit_t
            )
            names = ("v_xy", "v_scale", "v_rot", "v_colors", "v_emission", "v_sharpness", "v_softness")
        else:
            grads = cuda.nd_rasterize_backward_topk_norm_hexels(
                *bgeo, v_out, v_ws, v_ent, pixel_topk.contiguous(), aa, float(softness), sharp_t, emit_t
            )
            names = ("v_xy", "v_scale", "v_rot", "v_colors", "v_emission", "v_sharpness")
    elif tensor_softness:
        grads = cuda.nd_rasterize_backward_topk_norm_hexels_color_grad_softness(
            h, w, 16, 16, gaussian_ids_sorted, tile_bins, xys, scales, rotations, colors, color_grads, softness, 0.1,
            v_out, v_ws, v_ent, pixel_topk.contiguous(), aa, 0.0, sharp_t, emit_t,
        )  # fmt: skip
        names = ("v_xy", "v_scale", "v_rot", "v_colors", "v_color_grads", "v_emission", "v_sharpness", "v_softness")
    else:
        grads = cuda.nd_rasterize_backward_topk_norm_hexels_color_grad(
            h, w, 16, 16, gaussian_ids_sorted, tile_bins, xys, scales, rotations, colors, color_grads, 0.1,
            v_out, v_ws, v_ent, pixel_topk.contiguous(), aa, float(softness), sharp_t, emit_t,
        )  # fmt: skip
        names = ("v_xy", "v_scale", "v_rot", "v_colors", "v_color_grads", "v_emission", "v_sharpness")
    raster = {
        "xys": xys, "scales": scales, "rotations": rotations, "colors": colors, "color_grads": color_grads,
        "softness": softness if tensor_softness else torch.tensor([softness]),
        "softness_is_tensor": torch.tensor([int(tensor_softness)], dtype=torch.int32),
        "color_grad": torch.tensor([int(color_grad)], dtype=torch.int32),
        "size": torch.tensor([h, w], dtype=torch.int32), "aa_kernel_size": torch.tensor([aa]),
        "gaussian_ids_sorted": gaussian_ids_sorted.int(), "tile_bins": tile_bins.int(),
        "out_img": out_img, "pixel_topk": pixel_topk.int(), "weights_sum": weights_sum, "entropy": entropy,
        "v_out": v_out, "v_weights_sum": v_ws, "v_entropy": v_ent,
    }  # fmt: skip
    if sharpness is not None:
        raster["sharpness"] = sharpness
    if emission is not None:
        raster["emission"] = emission
    for nm, gr in zip(names, grads, strict=True):
        if gr.numel():
            raster[nm] = gr
    binning = {"xys": xys, "radii": radii.int(), "cum_tiles_hit": cum_tiles_hit.int(), "isect_ids": isect_ids, "gaussian_ids": gaussian_ids.int(),
               "isect_ids_sorted": isect_sorted, "tile_bins": tile_bins.int(), "tile_bounds": torch.tensor(tile_bounds, dtype=torch.int32)}  # fmt: skip
    return {f"/raster2d/{name}": raster, f"/binning/{name}": binning}


def _gaussian_case(utils, cuda, name, n, h, w, seed) -> dict[str, dict[str, Tensor]]:
    """Image-GS's projection, binning and top-K rasterizer, forward and backward (`/gaussians2d/<case>`)."""
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 2, generator=g).float().cuda()
    scales = (0.5 + 4.0 * torch.rand(n, 2, generator=g)).float()
    scales[:5] = 0.0  # zero determinant: projected to nothing
    scales = scales.cuda()
    rotation = ((torch.rand(n, 1, generator=g) - 0.5) * 6.283).float().cuda()
    colors = torch.rand(n, 3, generator=g).float().cuda()
    tile_bounds = ((w + 15) // 16, (h + 15) // 16, 1)
    xys, radii, conics, num_tiles_hit = cuda.project_gaussians_2d_scale_rot_forward(n, means, scales, rotation, h, w, tile_bounds)
    num_intersects, cum_tiles_hit = utils.compute_cumulative_intersects(num_tiles_hit)
    _, _, _, ids_sorted, tile_bins = utils.bin_and_sort_gaussians(n, num_intersects, xys, radii, cum_tiles_hit, tile_bounds)
    out_img, pixel_topk = cuda.nd_rasterize_forward_topk_norm(tile_bounds, (16, 16, 1), (w, h, 1), ids_sorted, tile_bins, xys, conics, colors)
    v_out = torch.randn(h, w, 3, generator=g).float().cuda()
    v_xy, v_conic, v_colors = cuda.nd_rasterize_backward_topk_norm(h, w, 16, 16, ids_sorted, tile_bins, xys, conics, colors, v_out, pixel_topk)
    v_xys_in = torch.randn(n, 2, generator=g).float().cuda()
    v_conics_in = torch.randn(n, 3, generator=g).float().cuda()
    v_cov2d, v_mean, v_scale, v_rot = cuda.project_gaussians_2d_scale_rot_backward(
        n, means, scales, rotation, h, w, radii, conics, v_xys_in, v_conics_in
    )
    record = {
        "means": means, "scales": scales, "rotation": rotation, "colors": colors, "size": torch.tensor([h, w], dtype=torch.int32),
        "xys": xys, "radii": radii.int(), "conics": conics, "num_tiles_hit": num_tiles_hit.int(),
        "gaussian_ids_sorted": ids_sorted.int(), "tile_bins": tile_bins.int(), "out_img": out_img, "pixel_topk": pixel_topk.int(),
        "v_out": v_out, "v_xy": v_xy, "v_conic": v_conic, "v_colors": v_colors,
        "v_xys_in": v_xys_in, "v_conics_in": v_conics_in, "v_cov2d": v_cov2d, "v_mean": v_mean, "v_scale": v_scale, "v_rot": v_rot,
    }  # fmt: skip
    return {f"/gaussians2d/{name}": record}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="hexels-src checkout ($PREFIX/share/hexels-src)")
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    import gsplat.cuda  # pyrefly: ignore[missing-import]
    import gsplat.utils  # pyrefly: ignore[missing-import]
    import gsplat_hexels as gh  # pyrefly: ignore[missing-import]

    records: dict[str, dict[str, Tensor]] = {}
    for case in CASES:
        records |= _case(gh, gsplat.utils, gsplat.cuda, *case)
    records |= _gaussian_case(gsplat.utils, gsplat.cuda, "imagegs", 1000, 96, 112, 20)
    records |= _gaussian_case(gsplat.utils, gsplat.cuda, "small", 300, 48, 64, 21)
    existing = load()
    rec = rr.RecordingStream("hexels_mojo_references")
    rec.save(str(REFERENCE_FILE))
    for path, r in existing.items():
        if not path.startswith(("/raster2d/", "/binning/", "/gaussians2d/")):
            rec.log(path, rr.AnyValues(**encode(r.tensors | r.texts)), static=True)
    for path, values in records.items():
        rec.log(path, rr.AnyValues(**encode({k: v.cpu() for k, v in values.items()})), static=True)
    rec.flush()
    print(f"wrote {len(records)} new records to {REFERENCE_FILE}")


if __name__ == "__main__":
    main()
