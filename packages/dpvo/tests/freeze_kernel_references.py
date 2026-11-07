"""Freeze the CUDA extension's outputs on real DPVO inputs into `fixtures/kernel_references.rrd`.

Runs DPVO (fast preset, stride 2) on the first frames of `data/iphone/pool.mp4` with the CUDA
extensions `_cuda_corr` / `_cuda_ba`, captures the arguments of one call of each kernel once the
patch graph is full, cuts each capture down to a few MB (a spatial crop of the feature maps,
the first frames' patch slots for BA), runs the CUDA kernel again on the cut inputs
and stores inputs and outputs. The cut keeps the arithmetic of every stored output: the
crop is applied before the reference runs, and index remapping is monotonic, so sorted-unique
orders are unchanged.

Run on a CUDA host: `pixi run -e dpvo-dev dpvo-freeze-kernel-references`. Re-run only when the
kernels' contract or the capture changes.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rerun as rr
import torch
from kernel_references import REFERENCE_FILE, encode  # pyrefly: ignore[missing-import]
from torch import Tensor

import dpvo.fastba
import dpvo.fastba.ba
from dpvo import _cuda_ba, _cuda_corr
from dpvo.api.inference import run_dpvo_pipeline
from dpvo.config import DPVOConfig

VIDEO: Path = Path("data/iphone/pool.mp4")
MAX_FRAMES: int = 240
# Capture the n-th call of each function (counted from 1): late enough that the patch graph is full.
CAPTURE_AT: dict[str, int] = {"corr": 300, "patchify": 400, "ba": 150, "neighbors": 150}
SEED: int = 1234
MAX_CORR_EDGES: int = 256
BA_REPEATS: int = 6
# BA's Hessian kernel switches on float32 comparisons (Z > 0.2, |r| < 128, the image band). An edge
# this close to a switch flips with the FMA placement (it did between CUDA and the arm64 CPU build at
# Z = 0.199991): its weight is set to 0 in the stored case, so both sides drop it.
THRESHOLD_MARGIN: dict[str, float] = {"depth": 1e-3, "pixels": 1e-2}

captured: dict[str, list[dict[str, Any]]] = {name: [] for name in CAPTURE_AT}
calls: dict[str, int] = dict.fromkeys(CAPTURE_AT, 0)


def _snapshot(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(a.detach().clone() if isinstance(a, Tensor) else a for a in args)


def _wrap(name: str, fn: Callable[..., Any], count: int = 1) -> Callable[..., Any]:
    """Record the inputs (and, after the call, the in-place results) of calls CAPTURE_AT[name] .. +count-1."""

    def wrapped(*args: Any) -> Any:
        calls[name] += 1
        take: bool = CAPTURE_AT[name] <= calls[name] < CAPTURE_AT[name] + count
        before: tuple[Any, ...] = _snapshot(args) if take else ()
        out = fn(*args)
        if take:
            captured[name].append({"args": before, "after": _snapshot(args), "out": out})
        return out

    return wrapped


def _crop_box(xy: Tensor, size: int, height: int, width: int) -> tuple[int, int]:
    """Top-left (x0, y0) of a size x size window inside the map, centred on a point of `xy` (n, 2) that lies in the map."""
    in_map: Tensor = xy[(xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)]
    assert in_map.shape[0] > 0, "no point inside the map"
    cx, cy = in_map[in_map.shape[0] // 2].tolist()
    x0: int = int(min(max(round(cx) - size // 2, 0), max(width - size, 0)))
    y0: int = int(min(max(round(cy) - size // 2, 0), max(height - size, 0)))
    return x0, y0


def _remap(index: Tensor) -> tuple[Tensor, Tensor]:
    """(sorted unique values, index re-expressed as positions in them): a monotonic re-indexing."""
    uniq, inverse = torch.unique(index, sorted=True, return_inverse=True)
    return uniq, inverse


def corr_case(fmap1: Tensor, fmap2: Tensor, coords: Tensor, ii: Tensor, jj: Tensor, radius: int, size: int) -> dict[str, Tensor]:
    """Edges into the busiest target frame whose patch centre lies in a size x size crop (+ a 6 px band past it)."""
    frame: int = int(jj.mode().values)
    centre: Tensor = coords[0, :, :, coords.shape[-2] // 2, coords.shape[-1] // 2]  # (E, 2) = (x, y)
    x0, y0 = _crop_box(centre[jj == frame], size, fmap2.shape[-2], fmap2.shape[-1])
    inside: Tensor = (centre[:, 0] >= x0 - 6) & (centre[:, 0] < x0 + size + 6) & (centre[:, 1] >= y0 - 6) & (centre[:, 1] < y0 + size + 6)
    sel: Tensor = (jj == frame) & inside
    assert int(sel.sum()) > 0, "empty correlation crop"
    picked: Tensor = sel.nonzero()[:, 0]
    sel = torch.zeros_like(sel)
    sel[picked[:: -(-picked.shape[0] // MAX_CORR_EDGES)]] = True  # every k-th edge: at most MAX_CORR_EDGES
    uniq_ii, ii_r = _remap(ii[sel])
    shift: Tensor = torch.tensor([x0, y0], dtype=coords.dtype, device=coords.device).view(1, 1, 2, 1, 1)
    case: dict[str, Tensor] = {
        "fmap1": fmap1[:, uniq_ii].contiguous(),
        "fmap2": fmap2[:, frame : frame + 1, :, y0 : y0 + size, x0 : x0 + size].contiguous(),
        "coords": (coords[:, sel] - shift).contiguous(),
        "ii": ii_r.contiguous(),
        "jj": torch.zeros_like(ii_r),
        "radius": torch.tensor([radius]),
    }
    case["out"] = _cuda_corr.forward(case["fmap1"], case["fmap2"], case["coords"], case["ii"], case["jj"], radius)[0]
    f32: dict[str, Tensor] = {"fmap1": case["fmap1"].float(), "fmap2": case["fmap2"].float()}
    case["out_float32"] = _cuda_corr.forward(f32["fmap1"], f32["fmap2"], case["coords"], case["ii"], case["jj"], radius)[0]
    return case


def patchify_case(net: Tensor, coords: Tensor, radius: int, size: int) -> dict[str, Tensor]:
    """Patch centres in a size x size crop of the first image (+ a 4 px band past it, for the bounds)."""
    xy: Tensor = coords[0]
    x0, y0 = _crop_box(xy, size, net.shape[-2], net.shape[-1])
    sel: Tensor = (xy[:, 0] >= x0 - 4) & (xy[:, 0] < x0 + size + 4) & (xy[:, 1] >= y0 - 4) & (xy[:, 1] < y0 + size + 4)
    assert int(sel.sum()) > 0, "empty patchify crop"
    shift: Tensor = torch.tensor([x0, y0], dtype=coords.dtype, device=coords.device)
    case: dict[str, Tensor] = {
        "net": net[:1, :, y0 : y0 + size, x0 : x0 + size].contiguous(),
        "coords": (coords[:1, sel] - shift).contiguous(),
        "radius": torch.tensor([radius]),
    }
    case["out"] = _cuda_corr.patchify_forward(case["net"], case["coords"], radius)[0]
    return case


def _near_threshold(poses: Tensor, patches: Tensor, intrinsics: Tensor, target: Tensor, ii: Tensor, jj: Tensor, kk: Tensor) -> Tensor:
    """Edges whose projected depth or pixel lies within THRESHOLD_MARGIN of a switch of the Hessian kernel (float64)."""
    from lietorch_mojo import SE3

    fx, fy, cx, cy = intrinsics.double().tolist()
    centre: Tensor = patches[kk][:, :, 1, 1].double()  # (E, 3): x, y, inverse depth
    point: Tensor = torch.stack([(centre[:, 0] - cx) / fx, (centre[:, 1] - cy) / fy, torch.ones_like(centre[:, 0]), centre[:, 2]], -1)
    G = SE3(poses.double()[jj]).mul(SE3(poses.double()[ii]).inv())
    xj = G.act(point)  # homogeneous (X, Y, Z, W)
    assert xj is not None
    z: Tensor = xj[:, 2]
    x1: Tensor = fx * xj[:, 0] / z + cx
    y1: Tensor = fy * xj[:, 1] / z + cy
    r: Tensor = torch.hypot(target[:, 0].double() - x1, target[:, 1].double() - y1)
    px: float = THRESHOLD_MARGIN["pixels"]
    near: Tensor = (z - 0.2).abs() < THRESHOLD_MARGIN["depth"]
    for value, edge in ((r, 128.0), (x1, -64.0), (y1, -64.0), (x1, 2 * cx + 64), (y1, 2 * cy + 64)):
        near |= (value - edge).abs() < px
    print(f"BA: {int(near.sum())} edges near a mask threshold get weight 0", file=sys.stderr)
    return near


def ba_case(args: tuple[Any, ...]) -> dict[str, Tensor]:
    """The BA call on the first n frames (the poses it touches and every patch slot of those frames).

    No re-indexing: `kk // ppf` stays the source frame, which the block-E path relies on, so the same
    inputs also give the block-E (`eff_impl`, global BA) reference.
    """
    poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, ppf, t0, t1, iterations, eff_impl = args
    assert not eff_impl, "capture a local BA call"
    n: int = int(torch.maximum(ii.max(), jj.max())) + 1
    weight = weight.clone()
    weight[0, _near_threshold(poses.view(-1, 7), patches.view(-1, *patches.shape[-3:]), intrinsics.view(-1, 4)[0], target[0], ii, jj, kk)] = 0.0
    flat: Tensor = patches.view(-1, *patches.shape[-3:])
    case: dict[str, Tensor] = {
        "poses": poses.view(-1, 7)[:n].clone().view(1, n, 7),
        "patches": flat[: n * ppf].clone().unsqueeze(0),
        "intrinsics": intrinsics.view(-1, 4)[:1].clone().view(1, 1, 4),
        "target": target.contiguous(),
        "weight": weight.contiguous(),
        "lmbda": lmbda.clone(),
        "ii": ii.clone(),
        "jj": jj.clone(),
        "kk": kk.clone(),
        "scalars": torch.tensor([ppf, t0, t1, iterations]),
    }
    for mode, eff in (("local", False), ("block_e", True)):
        # The float atomicAdds make CUDA itself nondeterministic: store run 0 and the largest distance of
        # BA_REPEATS - 1 more runs from it (relative to max(|ref|, 1)), which sets the tests' tolerance.
        runs: list[tuple[Tensor, Tensor]] = []
        for _ in range(BA_REPEATS):
            p, x = case["poses"].clone(), case["patches"].clone()
            no_prior: Tensor = torch.empty(0, dtype=torch.float32, device=p.device)
            _cuda_ba.forward(p, x, case["intrinsics"], case["target"], case["weight"], case["lmbda"], no_prior, no_prior, 0.0, 0.3,
                             case["ii"], case["jj"], case["kk"], ppf, t0, t1, iterations, eff)
            runs.append((p, x))
        for k, name in enumerate(("poses", "patches")):
            ref: Tensor = runs[0][k]
            case[f"{name}_{mode}"] = ref
            spread: float = max(((r[k] - ref).abs() / ref.abs().clamp_min(1.0)).max().item() for r in runs[1:])
            case[f"{name}_{mode}_spread"] = torch.tensor([spread], dtype=torch.float32)
    case["reproject_out"] = _cuda_ba.reproject(case["poses"].view(-1, 7), case["patches"].view(-1, 3, *patches.shape[-2:]), case["intrinsics"], case["ii"], case["jj"], case["kk"])
    return case


def main() -> None:
    torch.manual_seed(SEED)
    _cuda_corr.forward = _wrap("corr", _cuda_corr.forward, count=2)  # both pyramid levels of one update
    _cuda_corr.patchify_forward = _wrap("patchify", _cuda_corr.patchify_forward, count=4)  # imap, gmap, colours, grid
    _cuda_ba.forward = _wrap("ba", _cuda_ba.forward)
    dpvo.fastba.neighbors = dpvo.fastba.ba.neighbors = _wrap("neighbors", _cuda_ba.neighbors)
    rr.init("dpvo-freeze-kernel-references")  # the pipeline logs; nothing is saved
    for _ in run_dpvo_pipeline(dpvo_config=DPVOConfig.fast(), network_path="checkpoints/dpvo.pth", imagedir=str(VIDEO), stride=2, max_frames=MAX_FRAMES):
        pass
    missing: list[str] = [name for name, caps in captured.items() if not caps]
    assert not missing, f"no capture for {missing} (calls {calls}): lower CAPTURE_AT or raise MAX_FRAMES"

    records: dict[str, dict[str, Tensor]] = {}
    for level, (cap, size) in enumerate(zip(captured["corr"], (64, 24), strict=True)):
        fmap1, fmap2, coords, ii, jj, radius = cap["args"]
        records[f"/corr/level{level}"] = corr_case(fmap1, fmap2, coords, ii, jj, radius, size)
    for i, cap in enumerate(captured["patchify"]):
        net, coords, radius = cap["args"]
        records[f"/patchify/call{i}_r{radius}_{str(net.dtype).removeprefix('torch.')}"] = patchify_case(net, coords, radius, 48)
    records["/ba/window"] = ba_case(captured["ba"][0]["args"])
    kk, jj = captured["neighbors"][0]["args"]
    ix, jx = captured["neighbors"][0]["out"]
    records["/neighbors/graph"] = {"kk": kk, "jj": jj, "ix": ix, "jx": jx}

    rec = rr.RecordingStream("dpvo-kernel-references")
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    rec.save(str(REFERENCE_FILE))
    rec.send_property("source", rr.AnyValues(
        video=str(VIDEO), frames=str(MAX_FRAMES), seed=str(SEED), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        captured_calls=str(CAPTURE_AT),
    ))
    for entity, values in records.items():
        rec.log(entity, rr.AnyValues(**encode({k: v for k, v in values.items()})), static=True)
        sizes: str = ", ".join(f"{k}{tuple(v.shape)}" for k, v in values.items())
        print(f"{entity}: {sizes}", file=sys.stderr)
    rec.flush()
    print(f"wrote {REFERENCE_FILE} ({REFERENCE_FILE.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
