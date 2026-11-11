"""vipe_mojo against ViPE's CUDA extension (`vipe_ext`) on inputs frozen from one MetricTrail run on pool.mp4.

The references were frozen from vipe_ext (provenance in `vipe_references.py`). CPU runs in the unit lane;
CUDA is an integration test.

Tolerances: float32 outputs within 1e-5 of the reference, relative to max(|ref|, 1). Half-precision
`corr_index` within one float16 ulp (the Mojo kernel rounds every product and sum to half, as c10::Half
does). vipe_ext is built with `--use_fast_math` (approximate float division, sqrt, sin and cos), so no
port matches it bit for bit except where the kernel only multiplies and adds.

`ba_extended`: vipe_ext's BA has no float atomics, so repeated CUDA runs agree bit for bit and their
spread is no bound. The bound is BA_ULP_FACTOR times the largest distance between the reference and
vipe_ext's own results when the poses and disparities it updates move by at most one float32 ulp
(`*_ulp_spread`), or 1e-5 if larger. See README.md for the measured ratios that set the factor.
"""

import pytest

torch = pytest.importorskip("torch", reason="vipe_mojo requires PyTorch")
pytest.importorskip("rerun", reason="the references are an .rrd")
pytest.importorskip("vipe_mojo._vipe_mojo", reason="build the Mojo kernels first (pixi run -e metrictrail-dev vipe-mojo-build)")

from torch import Tensor  # noqa: E402
from vipe_references import REFERENCE_FILE, Record, entities, load  # noqa: E402  # pyrefly: ignore[missing-import]

import vipe_mojo  # noqa: E402
from vipe_mojo import droid_net_ext, slam_ext  # noqa: E402

if not REFERENCE_FILE.exists():
    pytest.skip(f"missing {REFERENCE_FILE.name} (git lfs pull)", allow_module_level=True)

DEVICES: list = [
    "cpu",
    pytest.param("cuda", marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]),
]
TOLERANCE: float = 1e-5
BA_ULP_FACTOR: float = 4.0
ENERGY_TOLERANCE: float = 1e-4
"""The per-iteration energy is a float32 sum over every pixel of every edge (vipe_ext sums it with torch.sum)."""


def _rel_error(ours: Tensor, ref: Tensor) -> float:
    ours_d, ref_d = ours.detach().cpu().double(), ref.double()
    return ((ours_d - ref_d).abs() / ref_d.abs().clamp_min(1.0)).max().item() if ref.numel() else 0.0


def _on(rec: Record, device: str, *names: str) -> list[Tensor]:
    """Copies on `device`: BA updates its inputs in place, and the records are cached across tests."""
    return [rec.tensor(n).to(device, copy=True) for n in names]


def test_module_surface() -> None:
    """`import vipe_mojo as _C` serves the six namespaces vipe.ext reads; unported functions raise when called."""
    for name in ("droid_net_ext", "grounding_dino_ext", "utils_ext", "slam_ext", "scatter_ext", "corr_ext"):
        assert hasattr(vipe_mojo, name), name
    with pytest.raises(NotImplementedError):
        vipe_mojo.utils_ext.compute_normal_weight()
    with pytest.raises(NotImplementedError):
        droid_net_ext.corr_index_backward()
    with pytest.raises(NotImplementedError):
        slam_ext.ba_extended_v2()
    with pytest.raises(NotImplementedError, match="vipe_ext.compute_normal_weight_robust"):
        vipe_mojo.compute_normal_weight_robust()  # vipe.utils.depth reads it from `vipe.ext._C`
    assert not hasattr(vipe_mojo.slam_ext, "_private")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("entity", entities("/corr_index"))
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_corr_index_matches_cuda(entity: str, dtype: str, device: str) -> None:
    rec: Record = load()[entity]
    volume, coords = _on(rec, device, "volume", "coords")
    ref: Tensor = rec.tensor("out")
    if dtype == "float32" and volume.dtype == torch.float16:
        volume, ref = volume.float(), rec.tensor("out_float32")
    elif dtype == "float16" and volume.dtype != torch.float16:
        pytest.skip("the captured volume is float32")
    (ours,) = droid_net_ext.corr_index_forward(volume, coords, int(rec.tensor("radius")))
    assert ours.shape == ref.shape and ours.dtype == ref.dtype
    if ref.dtype == torch.float16:
        ulp: Tensor = torch.finfo(torch.float16).eps * ref.float().abs().clamp_min(2.0**-14)
        diff: Tensor = (ours.cpu().float() - ref.float()).abs()
        assert (diff <= ulp).all(), f"max |diff| {diff.max():.3e}"
    else:
        assert _rel_error(ours, ref) <= TOLERANCE


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", ["captured", "staged", "float32"])
def test_altcorr_index_matches_cuda(path: str, device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The captured call (half features: the per-pixel fast path), the staged kernel on the same inputs, and float32."""
    rec: Record = load()["/altcorr_index/late"]
    fmap1, coords, ii, jj = _on(rec, device, "fmap1", "coords", "ii", "jj")
    pyramid: list[Tensor] = [rec.tensor(f"fmap2_{level}").to(device, copy=True) for level in range(4)]
    ref: Tensor = rec.tensor("out")
    if path != "captured":
        if fmap1.dtype != torch.float16:
            pytest.skip("the captured features are float32")
        if path == "staged":
            monkeypatch.setenv("VIPE_ALTCORR_PIX", "0")
            ref = rec.tensor("out_staged")
        else:
            fmap1, pyramid, ref = fmap1.float(), [p.float() for p in pyramid], rec.tensor("out_float32")
    (ours,) = droid_net_ext.altcorr_index_forward(fmap1, pyramid, coords, ii, jj, int(rec.tensor("radius")))
    assert ours.shape == ref.shape
    assert _rel_error(ours, ref) <= TOLERANCE


@pytest.mark.parametrize("device", DEVICES)
def test_frame_distance_matches_cuda(device: str) -> None:
    rec: Record = load()["/frame_distance/late"]
    poses, disps, intrinsics, pi, pj, qi, qj, di = _on(rec, device, "poses", "disps", "intrinsics", "pi", "pj", "qi", "qj", "di")
    ours: Tensor = slam_ext.frame_distance(poses, disps, intrinsics, pi, pj, qi, qj, di, float(rec.tensor("beta")))
    assert _rel_error(ours, rec.tensor("out")) <= TOLERANCE


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("entity", entities("/ba_extended"))
@pytest.mark.parametrize("_repeat", range(2))
@pytest.mark.parametrize("device_solve", [False, True])
def test_ba_extended_matches_cuda(entity: str, device: str, _repeat: int, device_solve: bool) -> None:
    """Repeated solves with fresh inputs must match the CUDA reference, including reused edge topology."""
    rec: Record = load()[entity]
    if device_solve and device != "cuda":
        pytest.skip("the device solver requires CUDA")
    poses, disps, intrinsics, disps_sens, targets, weights, eta, ii, jj, depth_active = _on(
        rec, device, "poses", "disps", "intrinsics", "disps_sens", "targets", "weights", "eta", "ii", "jj", "depth_active"
    )
    s: list[float] = rec.tensor("scalars").tolist()
    topology = slam_ext._prepare_topology(
        tuple(rec.tensor("ii").tolist()), tuple(rec.tensor("jj").tolist()), int(s[0]), int(s[1]),
        bool(s[7]), bool(s[5]), poses.device,
    )
    if device_solve:
        topology.pose_system.prepare_device(poses.device)
    _, _, energy = slam_ext.ba_extended(
        poses, disps, intrinsics, disps_sens, targets, weights, eta, ii, jj, depth_active,
        int(s[0]), int(s[1]), int(s[2]), s[3], s[4], bool(s[5]), s[6], bool(s[7]), s[8], s[9], s[10], bool(s[11]), s[12],
        topology=topology, device_solve=device_solve,
    )  # fmt: skip
    for name, ours in (("poses", poses), ("disps", disps), ("intrinsics", intrinsics)):
        floor: float = rec.tensor(f"{name}_ulp_spread").item()
        bound: float = max(TOLERANCE, BA_ULP_FACTOR * floor)
        error: float = _rel_error(ours, rec.tensor(f"{name}_out"))
        assert error <= bound, f"{name}: {error:.3e} (CUDA 1-ulp spread {floor:.3e})"
    if "energy_out" in rec.tensors:
        assert energy is not None
        assert _rel_error(energy, rec.tensor("energy_out")) <= ENERGY_TOLERANCE
