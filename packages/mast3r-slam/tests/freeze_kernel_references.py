"""Freeze the CUDA extension's outputs on real MASt3R-SLAM inputs into tests/fixtures/kernel_references.rrd.

Runs the 30-frame test configuration on normal-apt-tour.mp4 with the matching and Gauss-Newton calls routed to
`mast3r_slam._backends`, and keeps one late call of each: `iter_proj`, `refine_matches` and `gauss_newton_rays`. `gauss_newton_calib` never runs in that configuration,
so its record is the extension's calib solve on the captured rays problem, with the points constrained to the rays
of a pinhole K (fx = fy = 0.9 w, principal point at the centre), as `FactorGraph.solve_GN_calib` does.

usage (from packages/mast3r-slam, CUDA host): python tests/freeze_kernel_references.py
"""

import sys
import tempfile
from collections import Counter
from pathlib import Path

import rerun as rr
import torch
from torch import Tensor

import mast3r_slam.global_opt as global_opt
import mast3r_slam.matching as matching
from mast3r_slam import _backends as cuda
from mast3r_slam.api.inference import InferenceConfig, mast3r_slam_inference
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from simplecv.rerun_log_utils import RerunTyroConfig

sys.path.insert(0, str(Path(__file__).parent))
from kernel_references import REFERENCE_FILE, encode  # noqa: E402

CAPTURE_AT: dict[str, int] = {"iter_proj": 20, "refine_matches": 20, "gauss_newton_rays": 4}
"""Which call of each op to keep (late enough that the map has several keyframes)."""
QUERY_STRIDE: int = 6
"""The matching records keep every 6th query point (the ops treat each independently) and the whole images."""
GN_KEYFRAMES: int = 3
"""The Gauss-Newton records keep the first 3 keyframes of the captured problem and the edges between them."""
GN_RECORDS: Path = Path(tempfile.gettempdir()) / "mast3r-slam-freeze-gn-records.pt"
"""The backend process (spawned: it imports this module as __mp_main__, so the patches below apply there too) writes
its Gauss-Newton records here; the main process merges them."""
calls: Counter = Counter()
records: dict[str, dict[str, Tensor | str]] = {}
image_hw: list[int] = []


def _scalars(*values: float) -> Tensor:
    return torch.tensor(values, dtype=torch.float32)  # the extension takes them as C float


class CudaMatching:
    """`mast3r_slam._backends`'s matching ops, recording the CAPTURE_AT-th call of each."""

    @staticmethod
    def iter_proj(rays: Tensor, pts: Tensor, p_init: Tensor, max_iter: int, lambda_init: float, cost_thresh: float) -> list[Tensor]:
        calls["iter_proj"] += 1
        image_hw[:] = list(rays.shape[1:3])
        out: list[Tensor] = cuda.iter_proj(rays, pts, p_init, max_iter, lambda_init, cost_thresh)
        if calls["iter_proj"] == CAPTURE_AT["iter_proj"]:
            pts, p_init = pts[:, ::QUERY_STRIDE].contiguous(), p_init[:, ::QUERY_STRIDE].contiguous()
            p_out, valid_out = cuda.iter_proj(rays, pts, p_init, max_iter, lambda_init, cost_thresh)
            records["/iter_proj/late"] = {"rays": rays, "pts": pts, "p_init": p_init, "p_out": p_out, "valid_out": valid_out,
                                          "scalars": _scalars(max_iter, lambda_init, cost_thresh)}  # fmt: skip
        return out

    @staticmethod
    def refine_matches(D11: Tensor, D21: Tensor, p1: Tensor, radius: int, dilation_max: int) -> list[Tensor]:
        calls["refine_matches"] += 1
        out: list[Tensor] = cuda.refine_matches(D11, D21, p1, radius, dilation_max)
        if calls["refine_matches"] == CAPTURE_AT["refine_matches"]:
            D21, p1 = D21[:, ::QUERY_STRIDE].contiguous(), p1[:, ::QUERY_STRIDE].contiguous()
            (p_out,) = cuda.refine_matches(D11, D21, p1, radius, dilation_max)
            records["/refine_matches/late"] = {"D11": D11, "D21": D21, "p1": p1, "p_out": p_out, "scalars": _scalars(radius, dilation_max)}
        return out


def cuda_gn(name: str, *args: object) -> object:
    """`global_opt._call_gn_backend` on the CUDA extension, recording the CAPTURE_AT-th rays solve."""
    assert name == "gauss_newton_rays", name
    calls[name] += 1
    Twc, Xs, Cs, ii, jj, idx, valid, Q = (a.clone() for a in args[:8])  # type: ignore[union-attr]
    out = cuda.gauss_newton_rays(*args)
    if calls[name] == CAPTURE_AT[name]:
        # Twc / Xs / Cs rows follow the sorted keyframe ids of the edges; keep the first GN_KEYFRAMES and their edges.
        kept: Tensor = torch.unique(torch.cat([ii, jj]), sorted=True)[:GN_KEYFRAMES]
        edges: Tensor = torch.isin(ii, kept) & torch.isin(jj, kept)
        Twc, Xs, Cs = Twc[:GN_KEYFRAMES].clone(), Xs[:GN_KEYFRAMES].contiguous(), Cs[:GN_KEYFRAMES].contiguous()
        ii, jj, idx, valid, Q = (t[edges].contiguous() for t in (ii, jj, idx, valid, Q))
        assert torch.equal(torch.unique(torch.cat([ii, jj])), kept), (kept, ii, jj)
        problem: dict[str, Tensor] = {"Twc": Twc, "Xs": Xs, "Cs": Cs, "ii": ii, "jj": jj, "idx_ii2jj": idx, "valid_match": valid, "Q": Q}
        Twc_rays: Tensor = Twc.clone()
        (dx_rays,) = cuda.gauss_newton_rays(Twc_rays, Xs, Cs, ii, jj, idx, valid, Q, *args[8:])
        records["/gn_rays/late"] = problem | {"Twc_out": Twc_rays, "dx_out": dx_rays, "scalars": _scalars(*args[8:])}  # type: ignore[arg-type]
        cfg: dict = config["local_opt"]
        h, w = image_hw
        K: Tensor = torch.tensor([[0.9 * w, 0.0, w / 2], [0.0, 0.9 * w, h / 2], [0.0, 0.0, 1.0]], device=Xs.device)
        Xs_calib: Tensor = constrain_points_to_ray((h, w), Xs.view(-1, h, w, 3), K).view_as(Xs).contiguous()
        Twc_calib: Tensor = Twc.clone()
        scalars = (h, w, cfg["pixel_border"], cfg["depth_eps"], cfg["sigma_pixel"], cfg["sigma_depth"], cfg["C_conf"], cfg["Q_conf"],
                   cfg["max_iters"], cfg["delta_norm"])  # fmt: skip
        dx = cuda.gauss_newton_calib(Twc_calib, Xs_calib, Cs, K, ii, jj, idx, valid, Q, *scalars)
        # The rest of the problem (Twc, Cs, edges, matches, Q) is the rays record's.
        records["/gn_calib/late"] = {"problem": "/gn_rays/late", "Xs": Xs_calib, "K": K, "Twc_out": Twc_calib, "dx_out": dx[0],
                                     "scalars": _scalars(*scalars)}  # fmt: skip
        torch.save({e: {k: v.cpu() if isinstance(v, Tensor) else v for k, v in r.items()} for e, r in records.items() if e.startswith("/gn_")},
                   GN_RECORDS)  # fmt: skip
    return out


# Module level, so the spawned backend process gets them too.
matching._matching_backends = CudaMatching
global_opt._call_gn_backend = cuda_gn


def main() -> None:
    torch.manual_seed(0)
    GN_RECORDS.unlink(missing_ok=True)
    run_log: Path = Path(tempfile.gettempdir()) / "mast3r-slam-freeze-run.rrd"
    mast3r_slam_inference(
        InferenceConfig(rr_config=RerunTyroConfig(headless=True, save=run_log), dataset="data/normal-apt-tour.mp4",
                        config="config/test.yaml", no_viz=True, img_size=224, max_frames=30)  # fmt: skip
    )
    records.update(torch.load(GN_RECORDS))
    assert set(records) == {"/iter_proj/late", "/refine_matches/late", "/gn_rays/late", "/gn_calib/late"}, (sorted(records), calls)
    stream = rr.RecordingStream("mast3r-slam-kernel-references")
    stream.save(REFERENCE_FILE)
    for entity, values in records.items():
        stream.log(entity, rr.AnyValues(**encode(values)), static=True)
    stream.flush()
    print({e: {k: tuple(v.shape) for k, v in r.items() if isinstance(v, Tensor)} for e, r in records.items()}, dict(calls))


if __name__ == "__main__":
    main()
