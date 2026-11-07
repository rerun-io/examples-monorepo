"""The Mojo kernels (dpvo.mojo_backend) against the CUDA extension's outputs on real DPVO inputs.

The references are frozen from `_cuda_corr` / `_cuda_ba` on pool.mp4 (`freeze_kernel_references.py`),
so no test needs the CUDA extension. CPU runs in the unit lane; CUDA and MPS are integration tests.

Tolerances: float32 outputs within 1e-5 of the reference (relative to max(|ref|, 1)); float16
correlation within one float16 ulp of the reference value (the Mojo kernel rounds every product
and sum to half as c10::Half does, and the bilinear step is PyTorch on each device). BA compares
the poses and patch depths after the extension's Gauss-Newton iterations. Its float atomicAdds
make the CUDA extension itself nondeterministic, and the Mojo BA is one more run of the same sums in
another order. The bound is 1e-5 or a multiple of the largest distance of five more CUDA runs from
the stored one, whichever is larger: 3x for poses, 10x for patch depths. Weakly constrained depths
amplify the order noise, and the CPU path's chunked sums are further from GPU atomic order than two
GPU runs are (pool.mp4: depths up to 4.9x CUDA's spread of 3.9e-5; poses 3e-7).
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("rerun", reason="the references are an .rrd")
pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="build the Mojo kernels first (pixi run -e dpvo-dev dpvo-mojo-build)")

from kernel_references import REFERENCE_FILE, Record, entities, load  # noqa: E402  # pyrefly: ignore[missing-import]
from torch import Tensor  # noqa: E402

from dpvo.mojo_backend import ba, corr  # noqa: E402

if not REFERENCE_FILE.exists():
    pytest.skip(f"missing {REFERENCE_FILE.name}: run pixi run -e dpvo-dev dpvo-freeze-kernel-references on a CUDA host", allow_module_level=True)

DEVICES: list = [
    "cpu",
    pytest.param("cuda", marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]),
    pytest.param(
        "mps", marks=[pytest.mark.integration, pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")]
    ),
]
TOLERANCE: float = 1e-5
PRIOR_REFERENCE_FILE: Path = REFERENCE_FILE.parent / "metric_prior_references.rrd"
BA_SPREAD_FACTOR: dict[str, float] = {"poses": 3.0, "patches": 10.0}


def _rel_error(ours: Tensor, ref: Tensor) -> float:
    ours_d, ref_d = ours.detach().cpu().double(), ref.double()
    return ((ours_d - ref_d).abs() / ref_d.abs().clamp_min(1.0)).max().item() if ref.numel() else 0.0


def _on(rec: Record, device: str, *names: str) -> list[Tensor]:
    """Copies on `device`: BA updates its inputs in place, and the records are cached across tests."""
    return [rec.tensor(n).to(device, copy=True) for n in names]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("entity", entities("/corr"))
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_corr_matches_cuda(entity: str, dtype: str, device: str) -> None:
    rec: Record = load()[entity]
    fmap1, fmap2, coords, ii, jj = _on(rec, device, "fmap1", "fmap2", "coords", "ii", "jj")
    if dtype == "float32":
        fmap1, fmap2 = fmap1.float(), fmap2.float()
    (ours,) = corr.forward(fmap1, fmap2, coords, ii, jj, int(rec.tensor("radius")))
    ref: Tensor = rec.tensor("out" if dtype == "float16" else "out_float32")
    assert ours.shape == ref.shape and ours.dtype == ref.dtype
    if dtype == "float32":
        assert _rel_error(ours, ref) <= TOLERANCE
    else:
        ulp: Tensor = torch.finfo(torch.float16).eps * ref.float().abs().clamp_min(2.0**-14)
        assert ((ours.cpu().float() - ref.float()).abs() <= ulp).all(), f"max |diff| {(ours.cpu().float() - ref.float()).abs().max():.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("entity", entities("/patchify"))
def test_patchify_matches_cuda(entity: str, device: str) -> None:
    rec: Record = load()[entity]
    net, coords = _on(rec, device, "net", "coords")
    (ours,) = corr.patchify_forward(net, coords, int(rec.tensor("radius")))
    assert torch.equal(ours.cpu(), rec.tensor("out"))  # a copy: bit-exact


@pytest.mark.parametrize("device", DEVICES)
def test_reproject_matches_cuda(device: str) -> None:
    rec: Record = load()["/ba/window"]
    poses, patches, intrinsics, ii, jj, kk = _on(rec, device, "poses", "patches", "intrinsics", "ii", "jj", "kk")
    ours: Tensor = ba.reproject(poses, patches, intrinsics, ii, jj, kk)
    assert _rel_error(ours, rec.tensor("reproject_out")) <= TOLERANCE


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["local", "block_e"])
def test_ba_matches_cuda(mode: str, device: str) -> None:
    """One local-window BA call of a pool.mp4 run, solved densely (`local`) and with the block-E Schur (`block_e`)."""
    rec: Record = load()["/ba/window"]
    poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk = _on(
        rec, device, "poses", "patches", "intrinsics", "target", "weight", "lmbda", "ii", "jj", "kk"
    )
    ppf, t0, t1, iterations = rec.tensor("scalars").tolist()
    empty: Tensor = torch.empty(0, dtype=torch.float32, device=device)
    ba.forward(poses, patches, intrinsics, target, weight, lmbda, empty, empty, 0.0, 0.3, ii, jj, kk, ppf, t0, t1, iterations, mode == "block_e")
    for name, ours in (("poses", poses), ("patches", patches)):
        ref: Tensor = rec.tensor(f"{name}_{mode}")
        spread: float = rec.tensor(f"{name}_{mode}_spread").item()
        bound: float = max(TOLERANCE, BA_SPREAD_FACTOR[name] * spread)
        assert _rel_error(ours, ref) <= bound, f"{name}: {_rel_error(ours, ref):.3e} (CUDA repeat {spread:.3e})"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["local", "block_e"])
def test_ba_with_metric_prior_matches_cuda(mode: str, device: str) -> None:
    """One late BA call of MetricTrail on pool.mp4 with the metric depth prior (ViPE's DPVO extension), on both solve paths."""
    if not PRIOR_REFERENCE_FILE.exists():
        pytest.skip(f"missing {PRIOR_REFERENCE_FILE.name}: run pixi run -e metrictrail-dev dpvo-freeze-metric-prior-references on a CUDA host")
    rec: Record = load(PRIOR_REFERENCE_FILE)["/ba_prior/window"]
    poses, patches, intrinsics, target, weight, lmbda, prior, prior_weight, ii, jj, kk = _on(
        rec, device, "poses", "patches", "intrinsics", "target", "weight", "lmbda", "prior", "prior_weight", "ii", "jj", "kk"
    )
    ppf, t0, t1, iterations = rec.tensor("scalars").tolist()
    lam, huber = rec.tensor("prior_scalars").tolist()
    ba.forward(poses, patches, intrinsics, target, weight, lmbda, prior, prior_weight, lam, huber, ii, jj, kk, ppf, t0, t1, iterations, mode == "block_e")
    for name, ours in (("poses", poses), ("patches", patches)):
        ref: Tensor = rec.tensor(f"{name}_{mode}")
        spread: float = rec.tensor(f"{name}_{mode}_spread").item()
        bound: float = max(TOLERANCE, BA_SPREAD_FACTOR[name] * spread)
        assert _rel_error(ours, ref) <= bound, f"{name}: {_rel_error(ours, ref):.3e} (CUDA repeat {spread:.3e})"


@pytest.mark.parametrize("device", DEVICES)
def test_neighbors_matches_cuda(device: str) -> None:
    rec: Record = load()["/neighbors/graph"]
    kk, jj = _on(rec, device, "kk", "jj")
    ix, jx = ba.neighbors(kk, jj)
    assert torch.equal(ix.cpu(), rec.tensor("ix")) and torch.equal(jx.cpu(), rec.tensor("jx"))


def test_unported_paths_raise() -> None:
    empty: Tensor = torch.empty(0)
    with pytest.raises(NotImplementedError):
        corr.backward(empty, empty, empty, empty, empty, empty, 3)
    with pytest.raises(NotImplementedError):
        corr.patchify_backward(empty, empty, empty, 1)
    with pytest.raises(NotImplementedError):
        ba.solve_system(empty, empty, empty, empty, empty, 0.0, 0.0, 0)
