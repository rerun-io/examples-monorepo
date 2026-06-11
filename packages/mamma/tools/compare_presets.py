"""Side-by-side comparison of two dumped runs (e.g. quality vs fast) against the
original DAG, plus their wall times. Answers "what's the difference between the
presets" in one table.

  pixi run -e mamma-dev --frozen python tools/compare_presets.py \
      --quality-dir /tmp/run_quality --fast-dir /tmp/run_fast
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from jaxtyping import Float64
from numpy import ndarray


@dataclass
class CompareConfig:
    quality_dir: Path = Path("/tmp/qfdig/final_quality")
    """Dumped quality run (tools/dump_artifacts.py --preset quality)."""
    fast_dir: Path = Path("/tmp/qfdig/final_fast")
    """Dumped fast run."""
    golden_ma2d_dir: Path = Path(
        "/home/pablo/0Dev/repos/mamma/.claude/worktrees/baseline-3a4bc75/output/ma_2d/baseline-rj2/outdoors/running_jumping"
    )
    golden_ma3d_dir: Path = Path(
        "/home/pablo/0Dev/repos/mamma/.claude/worktrees/baseline-3a4bc75/output/ma_3d/baseline-rj2/outdoors/running_jumping"
    )
    golden_masks_npz: Path = Path("/tmp/qfdig/golden_masks_720p.npz")
    body_id: int = 0
    skip: int = 8


def _metrics(run_dir: Path, cfg: CompareConfig, gvj, gsm, g_masks, h_m: float) -> dict[str, float]:
    n_golden: int = gvj["pred_joints"].shape[0]
    frames: ndarray = np.arange(cfg.skip, n_golden)
    out: dict[str, float] = {}

    # masks mean IoU
    cm = np.load(run_dir / "masks.npz")
    cpos: dict[int, int] = {int(f): i for i, f in enumerate(cm["frame_indices"])}
    g_pk: ndarray = g_masks["packed"]
    c_pk: ndarray = cm["packed"]
    n_cams: int = g_pk.shape[1]
    hw: int = int(g_masks["mask_hw"][0]) * int(g_masks["mask_hw"][1])
    iou: list[float] = []
    for f in frames:
        if int(f) not in cpos:
            continue  # frame absent from this run's dump (subject not in all cameras that tick)
        cf: int = cpos[int(f)]
        for c in range(n_cams):
            a = np.unpackbits(g_pk[f, c])[:hw].astype(bool)
            b = np.unpackbits(c_pk[cf, c])[:hw].astype(bool)
            u = np.logical_or(a, b).sum()
            iou.append(1.0 if u == 0 else float(np.logical_and(a, b).sum()) / u)
    out["mask_mean_iou"] = float(np.mean(iou))

    # PVE + trans from fits
    fits = np.load(run_dir / "smplx_fits.npz")
    fpos: dict[int, int] = {int(f): i for i, f in enumerate(fits["frame_indices"])}
    common: list[int] = [f for f in range(cfg.skip, n_golden) if f in fpos]
    fr: ndarray = np.array([fpos[f] for f in common])
    o_v: Float64[ndarray, "F v 3"] = fits["vertices"][fr].astype(np.float64)
    g_v: Float64[ndarray, "F v 3"] = gvj["pred_vertices"][common].astype(np.float64)
    pve: ndarray = np.linalg.norm(o_v - g_v, axis=-1).mean(axis=1) * 1000.0
    out["pve_p95_mm"] = float(np.percentile(pve, 95))
    dtr: ndarray = np.linalg.norm(fits["trans"][fr].astype(np.float64) - gsm["smplx_translation"][common].astype(np.float64), axis=-1) * 1000.0
    out["trans_p95_mm"] = float(np.percentile(dtr, 95))

    timing = json.loads((run_dir / "timing.json").read_text())
    out["wall_s"] = float(timing["elapsed_s"])
    out["realtime_pct"] = 100.0 * float(timing["clip_seconds"]) / float(timing["elapsed_s"])
    out["sam2"] = timing.get("sam2_config", "?")
    out["tick_iters"] = timing.get("tick_iters", "?")
    return out


def main(cfg: CompareConfig) -> int:
    gvj = np.load(cfg.golden_ma3d_dir / f"verts_joints_body_id-{cfg.body_id:02d}.npz")
    gsm = np.load(cfg.golden_ma3d_dir / f"smplx_params_body_id-{cfg.body_id:02d}.npz", allow_pickle=True)
    g_masks = np.load(cfg.golden_masks_npz)
    ext: Float64[ndarray, "f 3"] = gvj["pred_joints"].max(axis=1) - gvj["pred_joints"].min(axis=1)
    h_m: float = float(np.median(np.linalg.norm(ext, axis=1)))

    q = _metrics(cfg.quality_dir, cfg, gvj, gsm, g_masks, h_m)
    f = _metrics(cfg.fast_dir, cfg, gvj, gsm, g_masks, h_m)

    print(f"\nQUALITY vs FAST  (vs original DAG on running_jumping; H={h_m:.3f} m)\n")
    print(f"  {'metric':<24}{'quality':>14}{'fast':>14}")
    print(f"  {'-' * 50}")
    def _model(s: object) -> str:
        name: str = str(s).split("/")[-1].replace(".yaml", "")
        return "sam2.1-small" if "hiera_s" in name else "etam-ti" if "efficienttam_ti" in name else name

    rows: list[tuple[str, str, str]] = [
        ("mask model", _model(q["sam2"]), _model(f["sam2"])),
        ("fit iters/tick", str(q["tick_iters"]), str(f["tick_iters"])),
        ("mask mean IoU", f"{q['mask_mean_iou']:.3f}", f"{f['mask_mean_iou']:.3f}"),
        ("mesh PVE p95 (mm)", f"{q['pve_p95_mm']:.1f}", f"{f['pve_p95_mm']:.1f}"),
        ("trans p95 (mm)", f"{q['trans_p95_mm']:.1f}", f"{f['trans_p95_mm']:.1f}"),
        ("wall (s)", f"{q['wall_s']:.1f}", f"{f['wall_s']:.1f}"),
        ("% of realtime", f"{q['realtime_pct']:.0f}%", f"{f['realtime_pct']:.0f}%"),
    ]
    for name, qv, fv in rows:
        print(f"  {name:<24}{qv:>14}{fv:>14}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(CompareConfig)))
