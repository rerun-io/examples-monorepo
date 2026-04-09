from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lietorch
import rerun as rr
import torch

from mast3r_slam.api.inference import SlamPipelineHandle, run_slam_pipeline
from mast3r_slam.config import load_config
from mast3r_slam.gn_backends import load_gn_backend
from mast3r_slam.mast3r_utils import load_mast3r

DEVICE = "cuda:0"
SYNTHETIC_ATOL = 1e-4
VIDEO_ATOL = 2e-3
POSE_RTOL = 1e-4
PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PKG_ROOT / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


@dataclass(frozen=True)
class BenchRow:
    name: str
    cuda_ms: float
    mojo_ms: float
    ratio: float
    max_abs: float
    same: bool


def _sync() -> None:
    torch.cuda.synchronize()


def _benchmark(fn, warmup: int = 5, runs: int = 20) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    samples: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        samples.append(start.elapsed_time(end))
    return float(torch.tensor(samples).median().item())


def _make_synthetic_inputs(
    n_poses: int = 8,
    h: int = 32,
    w: int = 32,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    hw = h * w
    twc = lietorch.Sim3.exp(0.02 * torch.randn(n_poses, 7, device=DEVICE)).data.contiguous().float()
    xs = torch.randn(n_poses, hw, 3, device=DEVICE, dtype=torch.float32)
    xs[..., 2].abs_().add_(1.0)
    cs = (0.5 + torch.rand(n_poses, hw, 1, device=DEVICE, dtype=torch.float32)).contiguous()
    ii_fwd = torch.arange(0, n_poses - 1, device=DEVICE, dtype=torch.long)
    jj_fwd = torch.arange(1, n_poses, device=DEVICE, dtype=torch.long)
    ii = torch.cat([ii_fwd, jj_fwd], dim=0).contiguous()
    jj = torch.cat([jj_fwd, ii_fwd], dim=0).contiguous()
    n_edges = int(ii.shape[0])
    idx = torch.arange(hw, device=DEVICE, dtype=torch.long).unsqueeze(0).repeat(n_edges, 1).contiguous()
    valid = torch.ones(n_edges, hw, 1, device=DEVICE, dtype=torch.bool).contiguous()
    q = (0.9 + 0.1 * torch.rand(n_edges, hw, 1, device=DEVICE, dtype=torch.float32)).contiguous()
    return (
        twc,
        xs,
        cs,
        ii,
        jj,
        idx,
        valid,
        q,
        0.5,
        0.25,
        0.1,
        0.1,
        5,
        1e-6,
    )


def _run_synthetic_backend(backend_name: str, args: tuple[torch.Tensor | float | int, ...]) -> torch.Tensor:
    backend = load_gn_backend(backend_name)
    twc = args[0].clone()
    backend.gauss_newton_rays(
        twc,
        *args[1:],
    )
    _sync()
    return twc


def bench_synthetic() -> BenchRow:
    args = _make_synthetic_inputs()
    cuda_pose = _run_synthetic_backend("cuda", args)
    mojo_pose = _run_synthetic_backend("mojo", args)
    cuda_ms = _benchmark(lambda: _run_synthetic_backend("cuda", args))
    mojo_ms = _benchmark(lambda: _run_synthetic_backend("mojo", args))
    max_abs = float((cuda_pose - mojo_pose).abs().max().item())
    same = bool(torch.allclose(cuda_pose, mojo_pose, atol=SYNTHETIC_ATOL, rtol=POSE_RTOL))
    return BenchRow("synthetic-rays", cuda_ms, mojo_ms, mojo_ms / cuda_ms, max_abs, same)


def _select_backend(name: str) -> None:
    os.environ["MAST3R_SLAM_GN_BACKEND"] = name
    import mast3r_slam.global_opt as global_opt

    global_opt._backends = load_gn_backend(name)


def _run_video_once(
    *,
    backend_name: str,
    dataset: str,
    config_path: str,
    img_size: int,
    max_frames: int,
    model,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    _select_backend(backend_name)
    load_config(config_path)
    rr.init(f"bench-{backend_name}", spawn=False)
    handle = SlamPipelineHandle()
    start = time.perf_counter()
    for _ in run_slam_pipeline(
        model=model,
        dataset_path=dataset,
        config_path=config_path,
        device=DEVICE,
        max_frames=max_frames,
        img_size=img_size,
        handle=handle,
    ):
        pass
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert handle.final_world_sim3_cam is not None
    assert handle.final_dataset_idx is not None
    return elapsed_ms, handle.final_world_sim3_cam, handle.final_dataset_idx


def bench_video(dataset: str, config_path: str, img_size: int, max_frames: int) -> BenchRow:
    model = load_mast3r(path=str(DEFAULT_WEIGHTS), device=DEVICE)
    model.share_memory()
    with contextlib.suppress(RuntimeError):
        torch.multiprocessing.set_start_method("spawn")
    cuda_ms, cuda_pose, cuda_idx = _run_video_once(
        backend_name="cuda",
        dataset=dataset,
        config_path=config_path,
        img_size=img_size,
        max_frames=max_frames,
        model=model,
    )
    mojo_ms, mojo_pose, mojo_idx = _run_video_once(
        backend_name="mojo",
        dataset=dataset,
        config_path=config_path,
        img_size=img_size,
        max_frames=max_frames,
        model=model,
    )
    cuda_map = {int(idx): pose for idx, pose in zip(cuda_idx.tolist(), cuda_pose, strict=False)}
    mojo_map = {int(idx): pose for idx, pose in zip(mojo_idx.tolist(), mojo_pose, strict=False)}
    common_idx = sorted(set(cuda_map) & set(mojo_map))
    cuda_pose = torch.stack([cuda_map[idx] for idx in common_idx], dim=0)
    mojo_pose = torch.stack([mojo_map[idx] for idx in common_idx], dim=0)
    max_abs = float((cuda_pose - mojo_pose).abs().max().item())
    same = bool(torch.allclose(cuda_pose, mojo_pose, atol=VIDEO_ATOL, rtol=POSE_RTOL))
    name = Path(dataset).stem
    return BenchRow(name, cuda_ms, mojo_ms, mojo_ms / cuda_ms, max_abs, same)


def _print_table(rows: list[BenchRow]) -> None:
    print("| benchmark | cuda ms | mojo ms | ratio | max abs diff | same |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row.name} | {row.cuda_ms:.3f} | {row.mojo_ms:.3f} | {row.ratio:.3f} | {row.max_abs:.2e} | {'yes' if row.same else 'no'} |"
        )


def main() -> None:
    rows = [
        bench_synthetic(),
        bench_video(str(PKG_ROOT / "data" / "normal-apt-tour.mp4"), str(PKG_ROOT / "config" / "base.yaml"), 512, 60),
        bench_video(str(PKG_ROOT / "data" / "livingroom-tour.mp4"), str(PKG_ROOT / "config" / "base.yaml"), 512, 60),
    ]
    _print_table(rows)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    main()
