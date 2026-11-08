"""hexels_mojo against the hexels-src CUDA extensions, on inputs and outputs frozen in fixtures/hexels_references.rrd.

The kernels are CUDA-only (as upstream), so every parity test is an integration test. Tolerances, relative
to max(|ref|, 1):
- prune scores 1e-4: upstream's scorer is built with `--use_fast_math` (approximate sin, cos, exp and
  division) and both accumulate with float atomics in scheduling order. The kept set (`prune_model`'s
  stable descending argsort) must agree except where two scores are within that tolerance.
- fused-ssim backward: bit-exact given upstream's derivative maps (multiplies and fma chains only).
- fused-ssim forward: map 1e-5, derivative maps 1e-3, mean SSIM 1e-6, end-to-end gradient 1e-4. Upstream
  is built without fast math, but nvcc and Mojo's LLVM fuse different multiply-adds in the windowed
  moments; sigma^2 = E[x^2] - mu^2 cancels on smooth images, and d map / d sigma1^2 = -C D / (A B^2)
  amplifies that most (measured 3.4e-4 on `/fused_ssim/smooth`; the map itself 5e-6).
- distCUDA2: bit-exact (the same Morton order, box pruning and fma chains; min/max and the stable sort are exact).
- 2D rasterizer (color-gradient family): tile binning and the forward outputs (image, top-10 ids, weights sum,
  entropy) bit-exact: the kernel issues nvcc's fast-math PTX instructions in nvcc's order (mojo/fastmath.mojo).
  Backward within 1e-4 of max(|ref|, 1): float atomics sum in scheduling order (measured <= 1e-5).
"""

import pytest

torch = pytest.importorskip("torch", reason="hexels_mojo requires PyTorch")
pytest.importorskip("rerun", reason="the references are an .rrd")

from hexels_references import REFERENCE_FILE, entities, load  # noqa: E402  # pyrefly: ignore[missing-import]
from torch import Tensor  # noqa: E402

if not REFERENCE_FILE.exists():
    pytest.skip(f"missing {REFERENCE_FILE.name}", allow_module_level=True)

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]

PRUNE_TOLERANCE: float = 1e-4
SSIM_MAP_TOLERANCE: float = 1e-5
SSIM_DERIVATIVE_TOLERANCE: float = 1e-3
SSIM_GRAD_TOLERANCE: float = 1e-4
RASTER_GRAD_TOLERANCE: float = 1e-4


def _rel_error(ours: Tensor, ref: Tensor) -> float:
    ours_d, ref_d = ours.detach().cpu().double(), ref.double()
    return ((ours_d - ref_d).abs() / ref_d.abs().clamp_min(1.0)).max().item()


@pytest.mark.parametrize("entity", entities("/prune"))
def test_removal_scores(entity: str) -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo.prune import removal_scores_cuda

    rec = load()[entity]
    xy, scales, rotations, colors, gradients, softness, image, target, selected = (
        rec.tensor(n).cuda() for n in ("xy", "scales", "rotations", "colors", "gradients", "softness", "image", "target", "selected")
    )
    scores, max_error = removal_scores_cuda(
        xy, scales, rotations, colors, gradients, softness, image, target, selected, bool(rec.tensor("rich").item())
    )
    ref_scores = rec.tensor("scores")
    assert _rel_error(scores, ref_scores) < PRUNE_TOLERANCE
    assert abs(max_error.item() - rec.tensor("max_error").item()) < PRUNE_TOLERANCE
    keep = len(ref_scores) // 2
    ours_kept = set(scores.cpu().argsort(descending=True, stable=True)[:keep].tolist())
    ref_kept = set(ref_scores.argsort(descending=True, stable=True)[:keep].tolist())
    boundary = ref_scores.sort(descending=True).values[keep - 1].item()
    disagree = {i for i in ours_kept ^ ref_kept if abs(ref_scores[i].item() - boundary) > PRUNE_TOLERANCE}
    assert not disagree, f"kept sets differ away from the boundary score: {sorted(disagree)}"


@pytest.mark.parametrize("entity", entities("/fused_ssim"))
def test_fused_ssim(entity: str) -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo._launch import launch
    from hexels_mojo.fused_ssim import C1, C2, fused_ssim

    rec = load()[entity]
    img1, img2, dl_dmap = (rec.tensor(n).cuda() for n in ("img1", "img2", "dl_dmap"))
    outputs = [torch.empty_like(img1) for _ in range(4)]
    launch("fused_ssim", (C1, C2, True), (img1, img2, *outputs))
    for name, ours in zip(("ssim_map", "dm_dmu1", "dm_dsigma1_sq", "dm_dsigma12"), outputs, strict=True):
        assert _rel_error(ours, rec.tensor(name)) < (SSIM_MAP_TOLERANCE if name == "ssim_map" else SSIM_DERIVATIVE_TOLERANCE), name

    ref_derivatives = [rec.tensor(n).cuda() for n in ("dm_dmu1", "dm_dsigma1_sq", "dm_dsigma12")]
    grad = torch.empty_like(img1)
    launch("fused_ssim_backward", (), (img1, img2, dl_dmap, grad, *ref_derivatives))
    assert torch.equal(grad.cpu(), rec.tensor("grad"))

    x = img1.clone().requires_grad_(True)
    loss = fused_ssim(x, img2)
    assert abs(loss.item() - rec.tensor("ssim_map").double().mean().item()) < 1e-6
    launch("fused_ssim_backward", (), (img1, img2, dl_dmap, grad, *outputs[1:]))
    assert _rel_error(grad, rec.tensor("grad")) < SSIM_GRAD_TOLERANCE


def test_fused_ssim_valid_padding_crops_the_map() -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo.fused_ssim import FusedSSIMMap, fused_ssim

    img1 = torch.rand(1, 3, 40, 50, device="cuda", requires_grad=True)
    img2 = torch.rand(1, 3, 40, 50, device="cuda")
    full = FusedSSIMMap.apply(0.01**2, 0.03**2, img1, img2, "same", True)
    valid = fused_ssim(img1, img2, padding="valid")
    assert torch.equal(valid, full[:, :, 5:-5, 5:-5].mean())
    valid.backward()
    assert img1.grad is not None and img1.grad.shape == img1.shape


@pytest.mark.parametrize("entity", entities("/distcuda2"))
def test_dist_cuda2(entity: str) -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo.simple_knn import distCUDA2

    rec = load()[entity]
    assert torch.equal(distCUDA2(rec.tensor("points").cuda()).cpu(), rec.tensor("dists"))


@pytest.mark.parametrize("entity", entities("/binning"))
def test_tile_binning(entity: str) -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo.raster2d import bin_and_sort_gaussians

    rec = load()[entity]
    xys, radii, cum = (rec.tensor(n).cuda() for n in ("xys", "radii", "cum_tiles_hit"))
    tile_bounds = tuple(rec.tensor("tile_bounds").tolist())
    isect, ids, isect_sorted, _, tile_bins = bin_and_sort_gaussians(xys.shape[0], rec.tensor("isect_ids").numel(), xys, radii, cum, tile_bounds)
    for name, ours in (("isect_ids", isect), ("gaussian_ids", ids), ("isect_ids_sorted", isect_sorted), ("tile_bins", tile_bins)):
        assert torch.equal(ours.cpu(), rec.tensor(name)), name


@pytest.mark.parametrize("entity", entities("/raster2d"))
def test_raster2d_color_grad(entity: str) -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo import raster2d

    rec = load()[entity]
    t = lambda name: rec.tensor(name).cuda()  # noqa: E731
    height, width = rec.tensor("size").tolist()
    aa = float(rec.tensor("aa_kernel_size").item())
    empty = torch.empty(0, device="cuda")
    sharpness = t("sharpness") if "sharpness" in rec.tensors else empty
    emission = t("emission") if "emission" in rec.tensors else empty
    per_hexel = bool(rec.tensor("softness_is_tensor").item())
    geometry = (t("xys"), t("scales"), t("rotations"), t("colors"), t("color_grads"))
    bins = (t("gaussian_ids_sorted"), t("tile_bins"))
    if per_hexel:
        outputs = raster2d.nd_rasterize_forward_topk_norm_hexels_color_grad_softness(
            (0, 0, 1), (16, 16, 1), (width, height, 1), *bins, *geometry, 0.1, aa, t("softness"), 0.0, sharpness, emission
        )
    else:
        outputs = raster2d.nd_rasterize_forward_topk_norm_hexels_color_grad(
            (0, 0, 1), (16, 16, 1), (width, height, 1), *bins, *geometry, 0.1, aa, float(rec.tensor("softness").item()), sharpness, emission
        )
    for name, ours in zip(("out_img", "pixel_topk", "weights_sum", "entropy"), outputs, strict=True):
        assert torch.equal(ours.cpu(), rec.tensor(name)), name

    grads_in = (t("v_out"), t("v_weights_sum"), t("v_entropy"), t("pixel_topk"), aa)
    if per_hexel:
        grads = raster2d.nd_rasterize_backward_topk_norm_hexels_color_grad_softness(
            height, width, 16, 16, *bins, *geometry, t("softness"), 0.1, *grads_in, 0.0, sharpness, emission
        )
        names = ("v_xy", "v_scale", "v_rot", "v_colors", "v_color_grads", "v_emission", "v_sharpness", "v_softness")
    else:
        grads = raster2d.nd_rasterize_backward_topk_norm_hexels_color_grad(
            height, width, 16, 16, *bins, *geometry, 0.1, *grads_in, float(rec.tensor("softness").item()), sharpness, emission
        )
        names = ("v_xy", "v_scale", "v_rot", "v_colors", "v_color_grads", "v_emission", "v_sharpness")
    for name, ours in zip(names, grads, strict=True):
        if name in rec.tensors:
            assert _rel_error(ours, rec.tensor(name)) < RASTER_GRAD_TOLERANCE, name


def test_overlay_falls_back_to_the_wrapped_module() -> None:
    from types import SimpleNamespace

    from hexels_mojo import raster2d

    wrapped = SimpleNamespace(other=lambda: "cuda", bin_and_sort_gaussians=lambda: "cuda")
    view = raster2d.overlay(wrapped)
    assert view.other() == "cuda"
    assert view.bin_and_sort_gaussians is raster2d.bin_and_sort_gaussians
