"""Gate a streaming-quality dump against the dataset's ``pred`` reference fits.

The MAMMA iPhone scenes ship per-subject ``pred/params_NN.npz`` (``mamma_pred``
final SMPL-X params: ``poses`` (F,165), ``betas`` (16,), ``trans`` (F,3)) and no
masks / 2D / triangulation — so this gate compares the SMPL-X BODY only. We
forward every ``pred`` subject through the SAME neutral SMPL-X the streaming
pipeline emits with, then for each of our emitted subjects compute per-frame PVE
vs pred.

Multi-person: our subjects (``smplx_fits_NN.npz``) are identity-matched to the
pred subjects (``params_NN.npz``) by minimal mean PVE (brute-force assignment;
subject counts are small), so the gate compares like-with-like regardless of
tracker ordering. Single-person falls back to ``smplx_fits.npz``.

PASS = every subject's per-frame PVE has p95 <= tol AND p99 <= tol (world frame,
no alignment), over emitted frames after ``skip_first_frames``, AND per-(cam x
people) realtime >= the speed floor.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from jaxtyping import Float
from numpy import ndarray

from mamma.fitting.smplx_wrapper import build_smplx_neutral, smplx_forward_per_parts

# Standard SMPL-X 165-dim axis-angle full-pose layout.
_GLOBAL = slice(0, 3)
_BODY = slice(3, 66)
_JAW = slice(66, 69)
_LHAND = slice(75, 120)
_RHAND = slice(120, 165)


def forward_params_to_verts(
    model,
    poses: Float[ndarray, "f 165"],
    betas: Float[ndarray, "nb"],
    trans: Float[ndarray, "f 3"],
    device: str,
    chunk: int = 64,
) -> Float[ndarray, "f 10475 3"]:
    """Forward flat 165-dim SMPL-X params through the neutral model, chunked."""
    n_frames: int = poses.shape[0]
    betas_t: Float[torch.Tensor, "1 nb"] = torch.as_tensor(betas.reshape(1, -1), dtype=torch.float32, device=device)
    verts_out: list[ndarray] = []
    for start in range(0, n_frames, chunk):
        end: int = min(start + chunk, n_frames)
        t_block: Float[torch.Tensor, "t 165"] = torch.as_tensor(poses[start:end].astype(np.float32), device=device)
        trans_block: Float[torch.Tensor, "t 3"] = torch.as_tensor(trans[start:end].astype(np.float32), device=device)
        with torch.inference_mode():
            out = smplx_forward_per_parts(
                model,
                global_orient=t_block[:, _GLOBAL],
                body_pose=t_block[:, _BODY],
                left_hand_pose=t_block[:, _LHAND],
                right_hand_pose=t_block[:, _RHAND],
                jaw_pose=t_block[:, _JAW],
                betas=betas_t,
                transl=trans_block,
            )
        verts_out.append(out.vertices.detach().cpu().numpy())
    return np.concatenate(verts_out, axis=0)


def _pve_series(
    our_verts: Float[ndarray, "f 10475 3"],
    our_frames: ndarray,
    ref_verts: Float[ndarray, "g 10475 3"],
    skip_first_frames: int,
) -> dict[int, float]:
    """Per-frame mean per-vertex L2 (mm), keyed by frame index. No alignment."""
    n_ref: int = ref_verts.shape[0]
    out: dict[int, float] = {}
    for i, f in enumerate(our_frames.tolist()):
        if f < skip_first_frames or f >= n_ref:
            continue
        out[f] = float(np.linalg.norm(our_verts[i] - ref_verts[f], axis=-1).mean()) * 1000.0
    return out


@dataclass
class ValidateVsPredConfig:
    run_dir: Path
    """Scene dump dir (smplx_fits[_NN].npz, timing.json, masks.npz)."""
    pred_dir: Path
    """Scene pred dir (params_NN.npz)."""
    n_people: int = 1
    """Number of subjects to gate (matched between our fits and pred)."""
    model_dir: Path = Path("data/body_models")
    """Body-model root containing ``smplx/SMPLX_NEUTRAL.npz``."""
    out_json: Path | None = None
    """Where to write the gate verdict (gate.json)."""
    skip_first_frames: int = 8
    """Drop the first N frames (bootstrap warmup) before scoring."""
    pve_p95_tol_mm: float = 30.0
    """PASS bound on the 95th-percentile per-frame PVE vs pred."""
    pve_p99_tol_mm: float = 30.0
    """PASS bound on the 99th-percentile per-frame PVE vs pred."""
    per_cam_realtime_floor: float = 0.25
    """PASS floor on per-(cam x people) realtime = n_cams * n_people * clip / wall."""
    min_coverage: float = 0.9
    """PASS floor on per-subject frame coverage = scored_frames / (pred_frames -
    skip_first_frames). Guards against a subject that drops most of the clip yet
    scores well on the few survivors (the dropout this sweep exists to catch)."""
    device: str = "cuda"
    """Compute device."""


def _load_our_subject(run_dir: Path, nn: int) -> dict | None:
    """Load our emitted subject nn (smplx_fits_NN.npz, or smplx_fits.npz for nn=0)."""
    p: Path = run_dir / f"smplx_fits_{nn:02d}.npz"
    if not p.exists() and nn == 0:
        p = run_dir / "smplx_fits.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {"frames": z["frame_indices"], "verts": z["vertices"].astype(np.float32)}


def main(config: ValidateVsPredConfig) -> int:
    model = build_smplx_neutral(config.model_dir, device=config.device)

    # --- Forward every pred subject; load every emitted subject -----------------
    pred_subj: list[Float[ndarray, "f 10475 3"]] = []
    for mm in range(config.n_people):
        pp = np.load(config.pred_dir / f"params_{mm:02d}.npz", allow_pickle=True)
        pred_subj.append(forward_params_to_verts(model, pp["poses"].astype(np.float32), pp["betas"].reshape(-1).astype(np.float32), pp["trans"].astype(np.float32), config.device))
    our_subj: list[dict | None] = [_load_our_subject(config.run_dir, nn) for nn in range(config.n_people)]

    n_emitted: int = sum(s is not None for s in our_subj)
    if n_emitted < config.n_people:
        print(f"FAIL: only {n_emitted}/{config.n_people} subjects emitted fits")
        if config.out_json is not None:
            config.out_json.parent.mkdir(parents=True, exist_ok=True)
            config.out_json.write_text(json.dumps({"scene": config.run_dir.name, "pass": False, "error": f"only_{n_emitted}_of_{config.n_people}_subjects"}, indent=2))
        return 1

    # --- Identity matching: assign our subjects to pred subjects (min total PVE) -
    cost: Float[ndarray, "n n"] = np.full((config.n_people, config.n_people), 1e9)
    pve_cache: dict[tuple[int, int], dict[int, float]] = {}
    for i, s in enumerate(our_subj):
        for j in range(config.n_people):
            series = _pve_series(s["verts"], s["frames"], pred_subj[j], config.skip_first_frames)
            pve_cache[i, j] = series
            if series:
                cost[i, j] = float(np.mean(list(series.values())))
    best_perm: tuple[int, ...] = min(itertools.permutations(range(config.n_people)), key=lambda perm: sum(cost[i, perm[i]] for i in range(config.n_people)))

    # --- Per-subject gate -------------------------------------------------------
    subjects: list[dict] = []
    accuracy_pass: bool = True
    for i in range(config.n_people):
        j: int = best_perm[i]
        series = pve_cache[i, j]
        if not series:
            accuracy_pass = False
            subjects.append({"our": i, "pred": j, "frames": 0, "pass": False})
            continue
        pve: ndarray = np.array([series[f] for f in sorted(series)])
        p95: float = float(np.percentile(pve, 95))
        p99: float = float(np.percentile(pve, 99))
        # Coverage: fraction of the reference (pred) frames we actually scored.
        # A low value means the subject was dropped for much of the clip — that
        # must FAIL even if the surviving frames look good.
        expected_frames: int = max(1, pred_subj[j].shape[0] - config.skip_first_frames)
        coverage: float = len(pve) / expected_frames
        ok: bool = p95 <= config.pve_p95_tol_mm and p99 <= config.pve_p99_tol_mm and coverage >= config.min_coverage
        accuracy_pass = accuracy_pass and ok
        worst: list[int] = sorted(series, key=lambda f: series[f], reverse=True)[:3]
        subjects.append({"our": i, "pred": j, "frames": len(pve), "coverage": coverage, "pve_mean_mm": float(pve.mean()), "pve_p95_mm": p95, "pve_p99_mm": p99, "pve_max_mm": float(pve.max()), "pass": ok, "worst_frames": {str(f): series[f] for f in worst}})
        print(f"  subj our{i}->pred{j}: {len(pve)}f cov {coverage:.0%}  PVE mean {pve.mean():5.1f} p95 {p95:5.1f} p99 {p99:5.1f} max {pve.max():5.1f}  {'PASS' if ok else 'FAIL'}")

    # --- Speed gate: per-(cam x people) realtime --------------------------------
    # Fail-closed: speed PASSES only with valid timing evidence. Missing/invalid
    # timing.json must not silently satisfy the documented realtime floor.
    per_cam_realtime: float | None = None
    speed_pass: bool = False
    timing_path: Path = config.run_dir / "timing.json"
    if timing_path.exists():
        timing: dict = json.loads(timing_path.read_text())
        elapsed_s: float = float(timing["elapsed_s"])
        clip_seconds: float = float(timing["clip_seconds"])
        masks_path: Path = config.run_dir / "masks.npz"
        n_cams: int = int(np.load(masks_path)["camera_names"].shape[0]) if masks_path.exists() else 4
        per_cam_realtime = n_cams * config.n_people * clip_seconds / elapsed_s if elapsed_s > 0 else 0.0
        speed_pass = per_cam_realtime >= config.per_cam_realtime_floor
        print(f"  speed: per-(cam x people) realtime {per_cam_realtime:.2f}x ({n_cams} cams x {config.n_people} ppl)  >= {config.per_cam_realtime_floor}: {'PASS' if speed_pass else 'FAIL'}")
    else:
        print(f"  speed: MISSING {timing_path.name} -> speed FAIL (cannot verify realtime floor)")

    overall: bool = accuracy_pass and speed_pass
    agg_p95: float = max((s.get("pve_p95_mm", 1e9) for s in subjects), default=1e9)
    agg_p99: float = max((s.get("pve_p99_mm", 1e9) for s in subjects), default=1e9)

    if config.out_json is not None:
        config.out_json.parent.mkdir(parents=True, exist_ok=True)
        config.out_json.write_text(json.dumps({
            "scene": config.run_dir.name,
            "n_people": config.n_people,
            "pve_p95_mm": agg_p95,  # worst subject (for the aggregate table)
            "pve_p99_mm": agg_p99,
            "pve_p95_tol_mm": config.pve_p95_tol_mm,
            "pve_p99_tol_mm": config.pve_p99_tol_mm,
            "accuracy_pass": accuracy_pass,
            "per_cam_realtime": per_cam_realtime,
            "speed_pass": speed_pass,
            "pass": overall,
            "subjects": subjects,
        }, indent=2))
        print(f"  wrote {config.out_json}")

    print(f"RESULT: {'PASS' if overall else 'FAIL'} (accuracy {'PASS' if accuracy_pass else 'FAIL'}, speed {'PASS' if speed_pass else 'FAIL'})")
    return 0 if overall else 1


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    sys.exit(main(tyro.cli(ValidateVsPredConfig)))
