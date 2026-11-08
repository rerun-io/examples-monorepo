"""hexels_mojo.raster3d (hexel-kernel's 3D hexel rasterizer in Mojo) against frozen CUDA outputs, stage by stage.

fixtures/raster3d_references.rrd (tests/freeze_raster3d_references.py, from `rasterize_hexels3d_forward_cached` and
`rasterize_hexels3d_backward_cached`): three configurations (linear color + view gradients + per-hexel softness; flat +
scalar softness + conservative bounds; quadratic-bounded color + specular lobe + per-edge softness). The forward is
bit-exact: the kernels issue nvcc's fast-math PTX in nvcc's order, and the stable sort reproduces cub's radix sort.
The backward scatters float atomics, so its gradients match to 1e-5 of each tensor's largest magnitude (two CUDA runs
differ by up to 4e-7 of it; the port by up to 1.1e-6).
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="hexels_mojo requires PyTorch")
pytest.importorskip("rerun", reason="the references are an .rrd")

from hexels_references import load  # noqa: E402  # pyrefly: ignore[missing-import]

RASTER3D_FILE: Path = Path(__file__).parent / "fixtures" / "raster3d_references.rrd"
if not RASTER3D_FILE.exists():
    pytest.skip(f"missing {RASTER3D_FILE.name}", allow_module_level=True)

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]

INPUTS = ("means3d", "scales", "rotations", "colors", "color_grads", "color_quads", "view_grads", "specular_ks", "specular_log_s",
          "opacities", "softnesses", "viewmatrix", "projmatrix", "campos", "background")  # fmt: skip
SAVED = ("radii_saved", "transmats", "opacities_out", "softnesses_out", "view_dirs", "normals", "points_xy_center", "ranges",
         "values_sorted", "final_transmittance", "last_index")  # fmt: skip
GRADS = ("means3d", "means2d", "scales", "rotations", "colors", "color_grads", "color_quads", "view_grads", "specular_ks",
         "specular_log_s", "opacities", "softnesses")  # fmt: skip
ENTITIES = list(load(RASTER3D_FILE))


def _args(tensors: dict) -> list:
    height, width, color_model, view_model, view_grads, conservative = tensors["config"].tolist()
    tan_x, tan_y, softness, modifier = tensors["camera"].tolist()
    inputs = [tensors[n].cuda().contiguous() if n in tensors else torch.empty(0, device="cuda") for n in INPUTS]
    return [*inputs, width, height, tan_x, tan_y, softness, modifier, color_model, view_model, bool(view_grads), False, bool(conservative), False]


@pytest.mark.parametrize("entity", ENTITIES)
def test_forward_is_bit_exact(entity: str) -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo.raster3d import rasterize_hexels3d_forward_cached

    tensors = load(RASTER3D_FILE)[entity].tensors
    out_color, radii, saved = rasterize_hexels3d_forward_cached(*_args(tensors))
    assert torch.equal(out_color.cpu(), tensors["out_color"])
    assert torch.equal(radii.cpu(), tensors["radii"])
    for name, ours in zip(SAVED, saved, strict=True):
        if name in tensors:
            assert torch.equal(ours.cpu(), tensors[name]), name
        else:
            assert ours.numel() == 0, name


@pytest.mark.parametrize("entity", ENTITIES)
def test_render_stage_is_bit_exact_from_cuda_intermediates(entity: str) -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo.raster3d import render

    tensors = load(RASTER3D_FILE)[entity].tensors
    height, width, color_model, view_model, view_grads, _ = tensors["config"].tolist()
    t = lambda name: tensors[name].cuda() if name in tensors else torch.empty(0, device="cuda")  # noqa: E731
    out_color, final_transmittance, last_index = render(
        t("ranges"), t("values_sorted"), t("colors"), t("color_grads"), t("color_quads"), t("view_grads"), t("specular_ks"),
        t("specular_log_s"), t("view_dirs"), t("normals"), t("transmats"), t("opacities_out"), t("softnesses_out"), t("background"),
        width, height, color_model, view_model, bool(view_grads),
    )  # fmt: skip
    assert torch.equal(out_color.cpu(), tensors["out_color"])
    assert torch.equal(final_transmittance.cpu(), tensors["final_transmittance"])
    assert torch.equal(last_index.cpu(), tensors["last_index"])


@pytest.mark.parametrize("entity", ENTITIES)
def test_cached_backward_matches_cuda(entity: str) -> None:
    pytest.importorskip("hexels_mojo._hexels_mojo", reason="build the Mojo kernels first (hexels-mojo-build)")
    from hexels_mojo.raster3d import rasterize_hexels3d_backward_cached

    tensors = load(RASTER3D_FILE)[entity].tensors
    args = _args(tensors)
    num_points: int = tensors["means3d"].shape[0]
    saved = [tensors[n].cuda() if n in tensors else torch.empty(0, device="cuda") for n in SAVED]
    empty = torch.empty(0, device="cuda")
    backward_args: list = [args[0], torch.zeros(num_points, 4, device="cuda"), *args[1:15], tensors["grad_out_color"].cuda(), empty, empty,
                           empty, empty, *args[15:24], -1, False, args[25], False, saved]  # fmt: skip
    grads = rasterize_hexels3d_backward_cached(*backward_args)
    for name, ours in zip(GRADS, grads, strict=True):
        key = f"bwd_{name}"
        if key not in tensors:
            assert ours.numel() == 0, name
            continue
        reference = tensors[key]
        bound = 1e-5 * max(1.0, float(reference.abs().max()))
        assert float((ours.cpu() - reference).abs().max()) <= bound, name


def test_overlay_keeps_prefiltering_and_backward_on_the_wrapped_module() -> None:
    from types import SimpleNamespace

    from hexels_mojo import raster3d

    calls: list[str] = []
    wrapped = SimpleNamespace(
        rasterize_hexels3d_forward=lambda *_: calls.append("cuda forward"),
        rasterize_hexels3d_backward=lambda *_: calls.append("cuda backward"),
        rasterize_hexels3d_backward_cached=lambda *_: calls.append("cuda cached backward"),
    )
    view = raster3d.overlay(wrapped)
    args: list[object] = [None] * 27
    args[raster3d.PREFILTER_ARG] = True
    view.rasterize_hexels3d_forward(*args)
    view.rasterize_hexels3d_backward()
    backward_args: list[object] = [torch.empty(0)] * 35
    backward_args[raster3d.BACKWARD_PREFILTER_ARG] = True
    view.rasterize_hexels3d_backward_cached(*backward_args)
    backward_args[raster3d.BACKWARD_PREFILTER_ARG] = False
    backward_args[18] = torch.ones(2, 2)  # a surface depth gradient
    view.rasterize_hexels3d_backward_cached(*backward_args)
    assert calls == ["cuda forward", "cuda backward", "cuda cached backward", "cuda cached backward"]
