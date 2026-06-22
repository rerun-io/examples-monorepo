"""Sweep the QUALITY preset over the iPhone scenes and gate each vs pred.

Serial (single GPU), resumable (skips any scene that already has a gate.json),
fault-tolerant (a scene that crashes or fails to bootstrap is marked FAILED and
the sweep continues). Per scene, composing the existing CLIs:

  1. dump_artifacts --preset quality --num-subjects N --rr-config.save -> quality.rrd + NPZs
  2. validate_vs_pred --n-people N                                      -> gate.json

PASS = PVE p95 <= 30 AND p99 <= 30 vs pred, and per-(cam x people) realtime >= 0.25x.
``subset`` picks single-person (pred has 1 params file), multi-person (>1), or all.
``--revalidate-only`` re-grades existing dumps with the current bound (no GPU dump).
Run from packages/mamma:  python tools/sweep_iphones_quality.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from tqdm import tqdm

# Scene categories (mirrors register_catalog.py's CATEGORIES — keep in sync).
CATEGORIES: tuple[str, ...] = ("indoors", "outdoors")


@dataclass
class SweepConfig:
    data_root: Path = Path("/mnt/8tb/data/mamma_markerless_iphones")
    """Root of the iPhone dataset (cat/scene/{meta,pred,videos_light})."""
    rrd_root: Path = Path("/mnt/8tb/data/mamma_markerless_iphones_rrds")
    """Output root; per scene -> <rrd_root>/<cat>/<scene>/{quality.rrd, *.npz, gate.json}."""
    trt_engine: Path = Path(".trt_cache/mammanet_b4_fp16_trt101339_sm120.plan")
    """MammaNet TensorRT engine (B=4)."""
    subset: Literal["all", "single", "multi"] = "all"
    """Which scenes: single-person, multi-person, or all 42."""
    force: bool = False
    """Re-run scenes that already have a gate.json (default: skip them / resume)."""
    revalidate_only: bool = False
    """Skip the GPU dump; just re-run validate_vs_pred on scenes with an existing dump."""
    scenes: list[str] | None = None
    """Explicit ``cat/scene`` list; default = the chosen subset."""
    report_only: bool = False
    """Skip all work; just aggregate existing gate.json files into the table."""


def list_scenes(data_root: Path, subset: str) -> list[tuple[str, int]]:
    """``(cat/scene, n_people)`` for the chosen subset (n_people = pred params count)."""
    out: list[tuple[str, int]] = []
    for cat in CATEGORIES:
        cat_dir: Path = data_root / cat
        if not cat_dir.is_dir():
            continue
        for sd in sorted(cat_dir.iterdir()):
            if not sd.is_dir():
                continue
            n_people: int = len(list((sd / "pred").glob("params_*.npz")))
            if n_people == 0:
                continue
            if subset == "single" and n_people != 1:
                continue
            if subset == "multi" and n_people <= 1:
                continue
            out.append((f"{cat}/{sd.name}", n_people))
    return out


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _validate(config: SweepConfig, scene: str, n_people: int, out_dir: Path, gate: Path) -> None:
    """Run validate_vs_pred (N-subject, identity-matched to pred)."""
    val = _run([
        sys.executable, "tools/validate_vs_pred.py",
        "--run-dir", str(out_dir),
        "--pred-dir", str(config.data_root / scene / "pred"),
        "--n-people", str(n_people),
        "--out-json", str(gate),
    ])
    if not gate.exists():
        tail: str = (val.stdout[-600:] + "\n" + val.stderr[-600:]).strip()
        gate.write_text(json.dumps({"scene": Path(scene).name, "pass": False, "error": "validate_crashed", "tail": tail}, indent=2))


def main(config: SweepConfig) -> int:
    scene_list: list[tuple[str, int]] = (
        [(s, len(list((config.data_root / s / "pred").glob("params_*.npz")))) for s in config.scenes]
        if config.scenes
        else list_scenes(config.data_root, config.subset)
    )
    n_pass: int = 0
    n_done: int = 0
    bar = tqdm(scene_list, desc=f"sweep[{config.subset}]", unit="scene", dynamic_ncols=True)
    for scene, n_people in bar:
        bar.set_postfix_str(f"{scene} ({n_people}p) | pass {n_pass}/{n_done}")
        out_dir: Path = config.rrd_root / scene
        gate: Path = out_dir / "gate.json"
        fits: Path = out_dir / "smplx_fits.npz"
        if config.report_only:
            continue
        if config.revalidate_only:
            if not fits.exists():
                tqdm.write(f"SKIP  {scene} (no dump to re-validate)")
                continue
            _validate(config, scene, n_people, out_dir, gate)
            v: dict = json.loads(gate.read_text())
            n_done += 1
            n_pass += int(bool(v.get("pass")))
            tqdm.write(f"{('PASS' if v.get('pass') else 'FAIL')}  {scene}  p95 {v.get('pve_p95_mm', 0):.1f} p99 {v.get('pve_p99_mm', 0) or 0:.1f}  per-camxppl {v.get('per_cam_realtime', 0) or 0:.2f}x")
            continue
        if gate.exists() and not config.force:
            tqdm.write(f"SKIP  {scene} (gate.json exists)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        dump = _run([
            sys.executable, "tools/dump_artifacts.py",
            "--preset", "quality", "--trt-engine", str(config.trt_engine),
            "--num-subjects", str(n_people),
            "--data-dir", str(config.data_root / scene), "--out-dir", str(out_dir),
            "--rr-config.save", str(out_dir / "quality.rrd"),
        ])
        n_done += 1
        if dump.returncode != 0 or not fits.exists():
            tail = (dump.stdout[-800:] + "\n" + dump.stderr[-800:]).strip()
            gate.write_text(json.dumps({"scene": Path(scene).name, "pass": False, "error": "dump_failed_or_incomplete", "tail": tail}, indent=2))
            tqdm.write(f"FAIL  {scene} (dump rc={dump.returncode} / no smplx_fits) — see gate.json")
            continue
        _validate(config, scene, n_people, out_dir, gate)
        v = json.loads(gate.read_text())
        n_pass += int(bool(v.get("pass")))
        tqdm.write(f"{'PASS' if v.get('pass') else 'FAIL'}  {scene}  p95 {v.get('pve_p95_mm', float('nan')):.1f} / p99 {v.get('pve_p99_mm', 0) or 0:.1f} mm  per-camxppl {v.get('per_cam_realtime', 0) or 0:.2f}x")

    # --- Aggregate report -------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"{'SCENE':38s} {'ppl':>3s} {'PVEp95':>7s} {'PVEp99':>7s} {'cxp-rt':>7s} {'ACC':>4s} {'SPD':>4s} {'PASS':>5s}")
    print("-" * 96)
    n_pass: int = 0
    n_total: int = 0
    failures: list[str] = []
    for scene, n_people in scene_list:
        gate = config.rrd_root / scene / "gate.json"
        if not gate.exists():
            print(f"{scene:38s} {n_people:>3d} {'--- not run ---':>40s}")
            continue
        n_total += 1
        v = json.loads(gate.read_text())
        if "error" in v:
            print(f"{scene:38s} {n_people:>3d}  ERROR: {v['error']}")
            failures.append(f"{scene} ({v['error']})")
            continue
        ok: bool = bool(v.get("pass"))
        n_pass += int(ok)
        if not ok:
            failures.append(f"{scene} (p95 {v.get('pve_p95_mm', 0):.0f} p99 {v.get('pve_p99_mm', 0) or 0:.0f} spd {v.get('per_cam_realtime') or 0:.2f})")
        pcr = v.get("per_cam_realtime") or 0.0
        print(f"{scene:38s} {n_people:>3d} {v.get('pve_p95_mm', 0):7.1f} {v.get('pve_p99_mm', 0) or 0:7.1f} {pcr:6.2f}x {'OK' if v.get('accuracy_pass') else 'X':>4s} {'OK' if v.get('speed_pass') else 'X':>4s} {'PASS' if ok else 'FAIL':>5s}")
    print("-" * 96)
    print(f"RESULT: {n_pass}/{n_total} scenes PASS (target {len(scene_list)}/{len(scene_list)})")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    return 0 if (n_pass == len(scene_list) and n_total == len(scene_list)) else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(SweepConfig)))
