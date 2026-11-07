"""Freeze ViPE's `vipe_ext` kernels into `fixtures/vipe_references.rrd` from one MetricTrail run on pool.mp4.

Runs MetricTrail (`metrictrail.apis.run.run_and_log`) in the metrictrail environment with the four
`vipe_ext` functions MetricTrail calls wrapped, and captures one late call of each:

- `droid_net_ext.corr_index_forward`: one full pyramid (four consecutive calls, levels 0-3), cut to one
  batch entry and a top-left window of query pixels (the volume's target grid stays whole).
- `droid_net_ext.altcorr_index_forward`: two edges, a top-left window of query pixels; the target
  frames keep their whole pyramids. Frames are re-indexed.
- `slam_ext.frame_distance`: up to 64 pairs, with the disparity frames they read re-indexed.
- `slam_ext.ba_extended`: the first call after `BA_CAPTURE_AFTER` for each option combination
  (motion_only, optimize_intrinsics, compute_energy). The frames it touches are re-indexed in order and
  the per-pixel inputs are cut to a top-left window, which is again a valid BA problem; that problem is
  solved BA_REPEATS times by `vipe_ext` for the reference and its repeat spread (zero: no float atomics), and
  BA_PERTURBED times with the poses and disparities moved by up to one float32 ulp, for the noise floor.

Every cut input is recomputed by `vipe_ext`, so each reference is the CUDA output for exactly the stored
inputs. Run on a CUDA host from this directory:
`pixi run -e metrictrail --frozen python freeze_vipe_references.py [scratch_dir]`.
"""

import contextlib
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import rerun as rr
import torch
import vipe.ext as ext
from metrictrail.apis.run import RunSettings, run_and_log
from torch import Tensor
from vipe_references import REFERENCE_FILE, encode  # pyrefly: ignore[missing-import]

VIDEO: Path = Path(__file__).resolve().parents[2] / "metrictrail" / "data" / "iphone" / "pool.mp4"
CORR_CAPTURE_AT: int = 7001  # of ~9,236 calls; 4k + 1 (one call per pyramid level, level 0 first)
ALTCORR_CAPTURE_AT: int = 500  # of ~606
FRAME_DISTANCE_CAPTURE_AT: int = 1000  # of ~1,266
BA_CAPTURE_AFTER: int = 1100  # of ~1,556
BA_STOP_AT: int = 1300
BA_REPEATS: int = 6
BA_PERTURBED: int = 6
SEED: int = 1234
CORR_WINDOW: tuple[int, int] = (12, 16)
ALTCORR_WINDOW: tuple[int, int] = (12, 16)
ALTCORR_EDGES: int = 2
FRAME_DISTANCE_PAIRS: int = 64
BA_WINDOW: tuple[int, int] = (16, 24)
BA_MAX_EDGES: int = 24

droid = ext.droid_net_ext
slam = ext.slam_ext
_corr_index = droid.corr_index_forward
_altcorr_index = droid.altcorr_index_forward
_frame_distance = slam.frame_distance
_ba_extended = slam.ba_extended

calls: Counter[str] = Counter()
ba_options: Counter[tuple[bool, bool, bool]] = Counter()
captured: dict[str, Any] = {}


def _clone(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple([t.detach().clone() for t in a] if isinstance(a, list) else a.detach().clone() if isinstance(a, Tensor) else a for a in args)


def corr_index(volume: Tensor, coords: Tensor, radius: int) -> list[Tensor]:
    calls["corr_index"] += 1
    level: int = calls["corr_index"] - CORR_CAPTURE_AT
    if 0 <= level < 4:
        captured[f"corr_index/level{level}"] = _clone((volume, coords, radius))
        print(f"captured corr_index call {calls['corr_index']}", file=sys.stderr)
    return _corr_index(volume, coords, radius)


def altcorr_index(fmap1: Tensor, pyramid: list[Tensor], coords: Tensor, ii: Tensor, jj: Tensor, radius: int) -> list[Tensor]:
    calls["altcorr_index"] += 1
    if calls["altcorr_index"] == ALTCORR_CAPTURE_AT:
        captured["altcorr_index"] = _clone((fmap1, pyramid, coords, ii, jj, radius))
    return _altcorr_index(fmap1, pyramid, coords, ii, jj, radius)


def frame_distance(*args: Any) -> Tensor:
    calls["frame_distance"] += 1
    if calls["frame_distance"] == FRAME_DISTANCE_CAPTURE_AT:
        captured["frame_distance"] = _clone(args)
    return _frame_distance(*args)


def ba_extended(*args: Any) -> list[Tensor]:
    calls["ba_extended"] += 1
    options: tuple[bool, bool, bool] = (bool(args[15]), bool(args[17]), bool(args[21]))
    ba_options[options] += 1
    key: str = "ba_extended/" + ("_".join(n for n, on in zip(("motion_only", "intrinsics", "energy"), options, strict=True) if on) or "full")
    capture: bool = calls["ba_extended"] >= BA_CAPTURE_AFTER and key not in captured
    before: tuple[Any, ...] = _clone(args) if capture else ()
    out: list[Tensor] = _ba_extended(*args)
    if capture:
        captured[key] = (before, _clone((args[0], args[1], args[2])), _clone(tuple(o for o in out if o is not None)))
        print(f"captured {key} at call {calls['ba_extended']}", file=sys.stderr)
    return out


class _Done(Exception):
    """Raised once every function is captured, to stop the run early."""


def _check_done() -> None:
    if calls["ba_extended"] >= BA_STOP_AT and all(k in captured for k in ("corr_index/level3", "altcorr_index", "frame_distance")):
        raise _Done


def _with_stop(fn: Any) -> Any:
    def wrapped(*args: Any) -> Any:
        out = fn(*args)
        _check_done()
        return out

    return wrapped


def _rel(a: Tensor, b: Tensor) -> float:
    return ((a.double() - b.double()).abs() / b.double().abs().clamp_min(1.0)).max().item() if b.numel() else 0.0


def corr_index_case(volume: Tensor, coords: Tensor, radius: int) -> dict[str, Tensor]:
    h, w = CORR_WINDOW
    volume_cut: Tensor = volume[:1, :h, :w].contiguous()
    coords_cut: Tensor = coords[:1, :, :h, :w].contiguous()
    (full,) = _corr_index(volume, coords, radius)
    (out,) = _corr_index(volume_cut, coords_cut, radius)
    assert torch.equal(out, full[:1, :, :, :h, :w]), "corr_index: the cut is not the full output's window"
    case: dict[str, Tensor] = {"volume": volume_cut, "coords": coords_cut, "radius": torch.tensor([radius]), "out": out}
    if volume.dtype == torch.float16:
        case["out_float32"] = _corr_index(volume_cut.float(), coords_cut, radius)[0]
    return case


def altcorr_index_case(fmap1: Tensor, pyramid: list[Tensor], coords: Tensor, ii: Tensor, jj: Tensor, radius: int) -> dict[str, Tensor]:
    h, w = ALTCORR_WINDOW
    m: int = ii.shape[0]
    edges: Tensor = torch.tensor(sorted({0, m // 2})[:ALTCORR_EDGES], device=ii.device)
    src, ii_cut = torch.unique(ii[edges], return_inverse=True)
    dst, jj_cut = torch.unique(jj[edges], return_inverse=True)
    fmap1_cut: Tensor = fmap1[:, src, :h, :w].contiguous()
    pyramid_cut: list[Tensor] = [p[:, dst].contiguous() for p in pyramid]
    coords_cut: Tensor = coords[:, edges, :h, :w].contiguous()
    (full,) = _altcorr_index(fmap1, pyramid, coords, ii, jj, radius)
    (out,) = _altcorr_index(fmap1_cut, pyramid_cut, coords_cut, ii_cut.contiguous(), jj_cut.contiguous(), radius)
    assert torch.equal(out, full[:, edges, :, :h, :w]), "altcorr_index: the cut is not the full output's window"
    case: dict[str, Tensor] = {"fmap1": fmap1_cut, "coords": coords_cut, "ii": ii_cut, "jj": jj_cut, "radius": torch.tensor([radius]), "out": out}
    case |= {f"fmap2_{level}": p for level, p in enumerate(pyramid_cut)}
    if fmap1.dtype == torch.float16:
        os.environ["VIPE_ALTCORR_PIX"] = "0"  # vipe_ext's staged kernel for half features
        case["out_staged"] = _altcorr_index(fmap1_cut, pyramid_cut, coords_cut, ii_cut, jj_cut, radius)[0]
        del os.environ["VIPE_ALTCORR_PIX"]
        case["out_float32"] = _altcorr_index(fmap1_cut.float(), [p.float() for p in pyramid_cut], coords_cut, ii_cut, jj_cut, radius)[0]
    return case


def frame_distance_case(
    poses: Tensor, disps: Tensor, intrinsics: Tensor, pi: Tensor, pj: Tensor, qi: Tensor, qj: Tensor, di: Tensor, beta: float
) -> dict[str, Tensor]:
    pairs: Tensor = torch.arange(min(FRAME_DISTANCE_PAIRS, pi.shape[0]), device=pi.device)
    frames, di_cut = torch.unique(di[pairs], return_inverse=True)
    full: Tensor = _frame_distance(poses, disps, intrinsics, pi, pj, qi, qj, di, beta)
    case: dict[str, Tensor] = {
        "poses": poses.contiguous(),
        "disps": disps[frames].contiguous(),
        "intrinsics": intrinsics.contiguous(),
        "pi": pi[pairs].contiguous(),
        "pj": pj[pairs].contiguous(),
        "qi": qi[pairs].contiguous(),
        "qj": qj[pairs].contiguous(),
        "di": di_cut.contiguous(),
        "beta": torch.tensor([beta], dtype=torch.float32),
    }
    case["out"] = _frame_distance(*(case[k] for k in ("poses", "disps", "intrinsics", "pi", "pj", "qi", "qj", "di")), beta)
    assert torch.equal(case["out"], full[pairs]), "frame_distance: the cut differs from the full output"
    return case


def _run_ba(case: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
    poses, disps, intrinsics = case["poses"].clone(), case["disps"].clone(), case["intrinsics"].clone()
    s: list[float] = case["scalars"].tolist()
    out: list[Tensor] = _ba_extended(
        poses, disps, intrinsics, case["disps_sens"], case["targets"], case["weights"], case["eta"], case["ii"], case["jj"], case["depth_active"],
        int(s[0]), int(s[1]), int(s[2]), s[3], s[4], bool(s[5]), s[6], bool(s[7]), s[8], s[9], s[10], bool(s[11]), s[12],
    )  # fmt: skip
    torch.cuda.synchronize()
    return poses, disps, intrinsics, out


def ba_extended_case(before: tuple[Any, ...], after: tuple[Tensor, Tensor, Tensor], key: str) -> dict[str, Tensor]:
    poses, disps, intrinsics, disps_sens, targets, weights, eta, ii, jj, depth_active = before[:10]
    t0, t1 = int(before[10]), int(before[11])
    # (t0, t1, iterations, lm, ep, motion_only, alpha, optimize_intrinsics, intrinsics_lm, intrinsics_ep, intrinsics_scale, compute_energy, flow_weight)
    scalars: Tensor = torch.tensor([float(x) for x in before[10:]], dtype=torch.float32)
    # Re-index the frames the call touches, in order: [t0, t1) stays contiguous, every ordering is kept.
    frames: Tensor = torch.unique(torch.cat([ii, jj, torch.arange(t0, t1, device=ii.device)]))
    remap: Tensor = torch.full((poses.shape[0],), -1, dtype=torch.int64, device=ii.device)
    remap[frames] = torch.arange(frames.shape[0], device=ii.device)
    kx_full: Tensor = torch.unique(torch.cat([torch.arange(t0, t1, device=ii.device), ii]))
    t0c: int = int(remap[t0])
    whole: dict[str, Tensor] = {
        "poses": poses[frames].contiguous(), "disps": disps[frames].contiguous(), "intrinsics": intrinsics.contiguous(),
        "disps_sens": disps_sens[frames].contiguous(), "targets": targets, "weights": weights, "eta": eta, "ii": remap[ii], "jj": remap[jj],
        "depth_active": depth_active[frames].contiguous(),
        "scalars": torch.cat([torch.tensor([t0c, t0c + t1 - t0], dtype=torch.float32), scalars[2:]]),
    }  # fmt: skip
    p, d, k, _ = _run_ba(whole)
    print(
        f"{key}: re-indexed whole call vs captured: poses {_rel(p, after[0][frames]):.2e}, disps {_rel(d, after[1][frames]):.2e}, "
        f"intrinsics {_rel(k, after[2]):.2e}; frames {frames.shape[0]}, edges {ii.shape[0]}, window [{t0}, {t1}), pixels {tuple(disps.shape[1:])}",
        file=sys.stderr,
    )
    # Cut: the first BA_MAX_EDGES edges and a top-left pixel window; keep only the frames they touch.
    h, w = BA_WINDOW
    e: int = min(BA_MAX_EDGES, ii.shape[0])
    ii_e, jj_e = ii[:e], jj[:e]
    frames = torch.unique(torch.cat([ii_e, jj_e, torch.arange(t0, t1, device=ii.device)]))
    remap.fill_(-1)
    remap[frames] = torch.arange(frames.shape[0], device=ii.device)
    kx_cut: Tensor = torch.unique(torch.cat([torch.arange(t0, t1, device=ii.device), ii_e]))
    eta_rows: Tensor = torch.searchsorted(kx_full, kx_cut)
    t0c = int(remap[t0])
    case: dict[str, Tensor] = {
        "poses": poses[frames].contiguous(),
        "disps": disps[frames, :h, :w].contiguous(),
        "intrinsics": intrinsics.contiguous(),
        "disps_sens": disps_sens[frames, :h, :w].contiguous(),
        "targets": targets[:e, :, :h, :w].contiguous(),
        "weights": weights[:e, :, :h, :w].contiguous(),
        "eta": eta[eta_rows, :h, :w].contiguous(),
        "ii": remap[ii_e].contiguous(),
        "jj": remap[jj_e].contiguous(),
        "depth_active": depth_active[frames].contiguous(),
        "scalars": torch.cat([torch.tensor([t0c, t0c + t1 - t0], dtype=torch.float32), scalars[2:]]),
    }
    runs: list[tuple[Tensor, Tensor, Tensor, list[Tensor]]] = [_run_ba(case) for _ in range(BA_REPEATS)]
    ref_p, ref_d, ref_k, ref_out = runs[0]
    case |= {"poses_out": ref_p, "disps_out": ref_d, "intrinsics_out": ref_k}
    if ref_out[2].numel():
        case["energy_out"] = ref_out[2]
    # vipe_ext's BA has no float atomics, so repeats agree bit for bit; the noise floor is how far CUDA moves when
    # the in-place inputs (poses, disparities) move by at most one float32 ulp.
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    perturbed: list[tuple[Tensor, Tensor, Tensor, list[Tensor]]] = []
    for _ in range(BA_PERTURBED):
        nudged: dict[str, Tensor] = dict(case)
        for name in ("poses", "disps"):
            ulps: Tensor = torch.randint(-1, 2, case[name].shape, generator=generator).to(case[name].device, torch.float32)
            nudged[name] = torch.nextafter(case[name], case[name] + ulps * torch.inf).where(ulps != 0, case[name])
        perturbed.append(_run_ba(nudged))
    repeated: list[tuple[Tensor, Tensor, Tensor]] = [(p, d, k) for p, d, k, _ in runs]
    nudged_out: list[tuple[Tensor, Tensor, Tensor]] = [(p, d, k) for p, d, k, _ in perturbed]
    for name, idx in (("poses", 0), ("disps", 1), ("intrinsics", 2)):
        case[f"{name}_spread"] = torch.tensor([max(_rel(r[idx], repeated[0][idx]) for r in repeated[1:])], dtype=torch.float32)
        case[f"{name}_ulp_spread"] = torch.tensor([max(_rel(r[idx], repeated[0][idx]) for r in nudged_out)], dtype=torch.float32)
    print(
        f"{key}: cut {e} edges, {frames.shape[0]} frames; CUDA repeat spread / 1-ulp input spread "
        + ", ".join(f"{n} {case[f'{n}_spread'].item():.2e} / {case[f'{n}_ulp_spread'].item():.2e}" for n in ("poses", "disps", "intrinsics")),
        file=sys.stderr,
    )
    return case


def main() -> None:
    scratch: Path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="vipe-freeze-"))  # ~100 MB of run outputs
    droid.corr_index_forward = _with_stop(corr_index)
    droid.altcorr_index_forward = _with_stop(altcorr_index)
    slam.frame_distance = _with_stop(frame_distance)
    slam.ba_extended = _with_stop(ba_extended)
    rec_run = rr.RecordingStream("vipe-freeze")
    rec_run.save(str(scratch / "run.rrd"))
    with contextlib.suppress(_Done):
        run_and_log(RunSettings(input=VIDEO, output=scratch / "out", capture=False), rec_run)
    droid.corr_index_forward, droid.altcorr_index_forward, slam.frame_distance, slam.ba_extended = (
        _corr_index,
        _altcorr_index,
        _frame_distance,
        _ba_extended,
    )
    print(f"calls: {dict(calls)}; ba options (motion_only, intrinsics, energy): {dict(ba_options)}", file=sys.stderr)
    for k, v in captured.items():
        args = v[0] if k.startswith("ba_extended") else v
        desc = [f"{tuple(a.shape)}:{str(a.dtype)[6:]}" if isinstance(a, Tensor) else f"[{len(a)}]" if isinstance(a, list) else repr(a) for a in args]
        print(f"captured {k}: {', '.join(desc)}", file=sys.stderr)

    cases: dict[str, dict[str, Tensor]] = {}
    for level in range(4):
        cases[f"/corr_index/level{level}"] = corr_index_case(*captured[f"corr_index/level{level}"])
    cases["/altcorr_index/late"] = altcorr_index_case(*captured["altcorr_index"])
    cases["/frame_distance/late"] = frame_distance_case(*captured["frame_distance"])
    for key in sorted(k for k in captured if k.startswith("ba_extended/")):
        before, after, _ = captured[key]
        cases[f"/{key}"] = ba_extended_case(before, after, key)

    rec = rr.RecordingStream("vipe-mojo-references")
    rec.save(str(REFERENCE_FILE))
    rec.send_property("source", rr.AnyValues(video=str(VIDEO.name), gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    for entity, case in cases.items():
        rec.log(entity, rr.AnyValues(**encode(case)), static=True)  # pyrefly: ignore[bad-argument-type]
        print(f"{entity}: " + ", ".join(f"{k}{tuple(v.shape)}:{str(v.dtype)[6:]}" for k, v in case.items()), file=sys.stderr)
    rec.flush()
    print(f"wrote {REFERENCE_FILE} ({REFERENCE_FILE.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
