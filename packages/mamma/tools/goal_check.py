"""Aggregate goal verification — runs every goal clause and prints PASS/FAIL.

Clauses:
  1. golden    — MPJPE/PVE vs golden ma_3d within tolerance (tools/validate_golden.py)
  2. realtime  — full clip <= 12.1 s wall incl. Rerun logging (tools/benchmark.py)
  3. datasets  — HOCap + Assembly101 demos run end-to-end and produce RRDs
  4. no-writes — streaming loop creates no files (tests/test_no_disk_writes.py)
  5. hygiene   — ruff + pyrefly + pytest clean in the mamma-dev env

Exit 0 only when all clauses PASS. Run from packages/mamma in the mamma env.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
PKG: Path = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path, tail: int = 6) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    lines: list[str] = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode == 0, "\n".join(lines[-tail:])


def main() -> int:
    py: str = sys.executable
    results: dict[str, bool] = {}

    print("=" * 70)
    engine = sorted(PKG.glob(".trt_cache/mammanet_*.plan"))
    engine_args: list[str] = ["--trt-engine", str(engine[-1])] if engine else []
    if not engine:
        print("note: no TRT engine in .trt_cache (run tools/build_trt_engine.py); using eager MammaNet")
    print("[1/5] golden gate (validate_golden)")
    ok, out = run([py, "tools/validate_golden.py", "--rr-config.headless", *engine_args], PKG)
    print(out)
    results["golden"] = ok

    print("=" * 70)
    print("[2/5] realtime benchmark (full clip, incl. Rerun logging)")
    ok, out = run([py, "tools/benchmark.py", "--rr-config.headless", *engine_args], PKG, tail=8)
    print(out)
    results["realtime"] = ok

    print("=" * 70)
    print("[3/5] dataset demos (HOCap + Assembly101 -> RRD)")
    with tempfile.TemporaryDirectory() as tmp:
        hocap_rrd: Path = Path(tmp) / "hocap.rrd"
        ok_h, out = run(
            [py, "tools/demo_hocap.py", "--max-frames", "60", "--rr-config.headless", "--rr-config.save", str(hocap_rrd)],
            PKG,
            tail=2,
        )
        print(out)
        ok_h = ok_h and hocap_rrd.exists() and hocap_rrd.stat().st_size > 100_000
        print(f"  hocap RRD: {'ok' if ok_h else 'MISSING/empty'}")
        assembly_rrd: Path = Path(tmp) / "assembly101.rrd"
        ok_a, out = run(
            [py, "tools/demo_assembly101.py", "--max-frames", "60", "--rr-config.headless", "--rr-config.save", str(assembly_rrd)],
            PKG,
            tail=2,
        )
        print(out)
        ok_a = ok_a and assembly_rrd.exists() and assembly_rrd.stat().st_size > 100_000
        print(f"  assembly101 RRD: {'ok' if ok_a else 'MISSING/empty'}")
    results["datasets"] = ok_h and ok_a

    print("=" * 70)
    print("[4/5] no-disk-writes (pytest tests/test_no_disk_writes.py)")
    # pytest lives in the dev env, not the prod env this script runs in.
    dev_python: Path = REPO_ROOT / ".pixi/envs/mamma-dev/bin/python"
    ok, out = run([str(dev_python), "-m", "pytest", "tests/test_no_disk_writes.py", "-q"], PKG, tail=3)
    print(out)
    results["no-writes"] = ok

    print("=" * 70)
    print("[5/5] hygiene (lint + typecheck + tests in mamma-dev)")
    hygiene: bool = True
    for task in ("lint", "typecheck", "tests"):
        ok, out = run(["pixi", "run", "-e", "mamma-dev", "--frozen", task], REPO_ROOT, tail=2)
        print(f"  {task}: {'PASS' if ok else 'FAIL'}\n    {out.splitlines()[-1] if out else ''}")
        hygiene = hygiene and ok
    results["hygiene"] = hygiene

    print("=" * 70)
    print("GOAL CHECK SUMMARY")
    for name, ok in results.items():
        print(f"  {name:<10} {'PASS' if ok else 'FAIL'}")
    overall: bool = all(results.values())
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
