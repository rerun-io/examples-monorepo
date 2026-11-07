"""Freeze MetricTrail's BA with the metric depth prior into `fixtures/metric_prior_references.rrd`.

This runs in the metrictrail environment: MetricTrail on the first frames of pool.mp4 (its DPVO is
dpvo.metric on the CUDA extension `dpvo._cuda_ba`), capturing one late `forward` call of the CUDA
backend of `dpvo.kernels` with a positive prior weight. The case keeps the
first n frames' poses, patch slots and prior values (no re-indexing) and, for each solve path (dense
`local` and the block-E Schur `block_e`), the outputs of BA_REPEATS reruns of the CUDA call on them
(the float atomics make CUDA nondeterministic; the tests bound the Mojo result by that spread).

Run on a CUDA host: `pixi run -e metrictrail-dev dpvo-freeze-metric-prior-references`.
"""

import contextlib
import sys
import tempfile
from pathlib import Path
from typing import Any

import rerun as rr
import torch
from kernel_references import encode  # pyrefly: ignore[missing-import]
from metrictrail.apis.run import RunSettings, run_and_log
from torch import Tensor

from dpvo import _cuda_ba, _cuda_corr, kernels

VIDEO: Path = Path("../metrictrail/data/iphone/pool.mp4")
OUTPUT: Path = Path(__file__).parent / "fixtures" / "metric_prior_references.rrd"
CAPTURE_AFTER: int = 400
BA_REPEATS: int = 6
SEED: int = 1234

captured: list[tuple[Any, ...]] = []
calls: list[int] = [0]
_forward = _cuda_ba.forward


class _Done(Exception):
    """Raised once the call is captured, to stop the run early."""


class _CapturingBA:
    """`dpvo._cuda_ba` as a `dpvo.kernels.BAKernels` whose `forward` records the first late call with a prior."""

    neighbors = staticmethod(_cuda_ba.neighbors)
    reproject = staticmethod(_cuda_ba.reproject)

    def forward(
        self, poses: Tensor, patches: Tensor, intrinsics: Tensor, target: Tensor, weight: Tensor, lmbda: Tensor, depth_prior: Tensor,
        depth_prior_weight: Tensor, depth_prior_lambda: float, depth_prior_huber: float, ii: Tensor, jj: Tensor, kk: Tensor, M: int,
        t0: int, t1: int, iterations: int, eff_impl: bool,
    ) -> list[Tensor]:  # fmt: skip
        args: tuple[Any, ...] = (poses, patches, intrinsics, target, weight, lmbda, depth_prior, depth_prior_weight, depth_prior_lambda,
                                 depth_prior_huber, ii, jj, kk, M, t0, t1, iterations, eff_impl)  # fmt: skip
        calls[0] += 1
        if calls[0] >= CAPTURE_AFTER and depth_prior_lambda > 0.0:
            captured.append(tuple(a.detach().clone() if isinstance(a, Tensor) else a for a in args))
            raise _Done
        return _forward(*args)


def _case(args: tuple[Any, ...]) -> dict[str, Tensor]:
    poses, patches, intrinsics, target, weight, lmbda, prior, prior_weight, lam, huber, ii, jj, kk, ppf, t0, t1, iterations, _ = args
    n: int = int(torch.maximum(ii.max(), jj.max())) + 1
    m: int = n * ppf
    case: dict[str, Tensor] = {
        "poses": poses.reshape(-1, 7)[:n].clone().view(1, n, 7),
        "patches": patches.reshape(-1, *patches.shape[-3:])[:m].clone().unsqueeze(0),
        "intrinsics": intrinsics.reshape(-1, 4)[:1].clone().view(1, 1, 4),
        "target": target.contiguous(),
        "weight": weight.contiguous(),
        "lmbda": lmbda.clone(),
        "prior": prior.reshape(-1)[:m].contiguous(),
        "prior_weight": prior_weight.reshape(-1)[:m].contiguous(),
        "ii": ii.clone(),
        "jj": jj.clone(),
        "kk": kk.clone(),
        "scalars": torch.tensor([ppf, t0, t1, iterations]),
        "prior_scalars": torch.tensor([lam, huber], dtype=torch.float32),
    }
    for mode in ("local", "block_e"):
        runs: list[tuple[Tensor, Tensor]] = []
        for _ in range(BA_REPEATS):
            p, x = case["poses"].clone(), case["patches"].clone()
            _forward(p, x, case["intrinsics"], case["target"], case["weight"], case["lmbda"], case["prior"], case["prior_weight"], lam, huber,
                     case["ii"], case["jj"], case["kk"], ppf, t0, t1, iterations, mode == "block_e")
            runs.append((p, x))
        for k, name in enumerate(("poses", "patches")):
            ref: Tensor = runs[0][k]
            case[f"{name}_{mode}"] = ref
            spread: float = max(((r[k] - ref).abs() / ref.abs().clamp_min(1.0)).max().item() for r in runs[1:])
            case[f"{name}_{mode}_spread"] = torch.tensor([spread], dtype=torch.float32)
    return case


def main() -> None:
    torch.manual_seed(SEED)
    kernels.register("cuda", corr=_cuda_corr, ba=_CapturingBA())  # the CUDA backend, with a capturing forward
    with tempfile.TemporaryDirectory(prefix="metric-prior-freeze-") as scratch:  # ~100 MB of run outputs, not kept
        rec_run = rr.RecordingStream("metric-prior-freeze")
        rec_run.save(str(Path(scratch) / "run.rrd"))
        with contextlib.suppress(_Done):
            run_and_log(RunSettings(input=VIDEO, output=Path(scratch) / "out", capture=False), rec_run)
    assert captured, f"no BA call with a depth prior after call {CAPTURE_AFTER} ({calls[0]} calls)"
    rec = rr.RecordingStream("dpvo-metric-prior-references")
    rec.save(str(OUTPUT))
    rec.send_property("source", rr.AnyValues(video=str(VIDEO), gpu=torch.cuda.get_device_name(), torch=torch.__version__, seed=str(SEED)))
    case: dict[str, Tensor] = _case(captured[0])
    rec.log("/ba_prior/window", rr.AnyValues(**encode(case)), static=True)
    print("/ba_prior/window: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in case.items()), file=sys.stderr)
    rec.flush()
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
