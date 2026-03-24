from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a combined COLMAP+images source tree and train a Brush splat."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Extracted image set root, e.g. .../images/.../0100")
    parser.add_argument("--model-dir", type=Path, required=True, help="COLMAP sparse model dir containing cameras.bin/images.bin/points3D.bin")
    parser.add_argument("--work-root", type=Path, required=True, help="Dataset-local Brush work directory")
    parser.add_argument("--steps", type=int, default=5000, help="Total Brush training steps")
    parser.add_argument("--eval-split-every", type=int, default=10, help="Hold out every nth image for Brush eval")
    parser.add_argument("--eval-every", type=int, default=1000, help="Run Brush eval every nth step")
    parser.add_argument("--export-every", type=int, help="Export every nth step; defaults to --steps")
    parser.add_argument("--experiment-name", default="5k", help="Run name under <work-root>/runs/")
    parser.add_argument("--with-viewer", action="store_true", help="Open the Brush viewer while training")
    return parser.parse_args()


def ensure_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"{label} does not exist or is not a directory: {resolved}")
    return resolved


def ensure_model_dir(path: Path) -> Path:
    resolved = ensure_dir(path, "model dir")
    missing = [name for name in ("cameras.bin", "images.bin", "points3D.bin") if not (resolved / name).is_file()]
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(f"model dir is missing required COLMAP files ({missing_str}): {resolved}")
    return resolved


def ensure_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if not link_path.is_symlink():
            raise SystemExit(f"expected symlink at {link_path}, found non-symlink")
        current = Path(os.readlink(link_path))
        current_resolved = (link_path.parent / current).resolve() if not current.is_absolute() else current.resolve()
        if current_resolved != target:
            raise SystemExit(
                f"existing symlink points somewhere else:\n  link: {link_path}\n  current: {current_resolved}\n  expected: {target}"
            )
        return
    link_path.symlink_to(target, target_is_directory=True)


def stream_process(cmd: list[str], log_path: Path, cwd: Path) -> None:
    print(f"Running Brush command:\n  {shlex.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(cmd)}\n")
        log_file.flush()
        start = time.monotonic()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
        elapsed = time.monotonic() - start
        summary = f"Brush run finished in {elapsed:.1f}s"
        if return_code != 0:
            log_file.write(f"{summary} (failed)\n")
            raise SystemExit(f"brush exited with status {return_code}; see {log_path}")
        sys.stdout.write(f"{summary}\n")
        log_file.write(f"{summary}\n")


def main() -> None:
    args = parse_args()

    brush_bin = shutil.which("brush") or shutil.which("brush_app")
    if brush_bin is None:
        raise SystemExit("brush/brush_app executable not found in PATH; install the pysfm pixi env first")

    images_dir = ensure_dir(args.images_dir, "images dir")
    model_dir = ensure_model_dir(args.model_dir)
    work_root = args.work_root.expanduser().resolve()
    experiment_name = args.experiment_name
    export_every = args.export_every or args.steps

    run_dir = work_root / "runs" / experiment_name
    if run_dir.exists():
        raise SystemExit(f"run directory already exists; refusing to overwrite: {run_dir}")

    source_dir = work_root / "source"
    source_sparse_dir = source_dir / "sparse"
    source_sparse_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)

    ensure_symlink(source_dir / "images", images_dir)
    ensure_symlink(source_sparse_dir / "0", model_dir)

    cmd = [
        brush_bin,
        str(source_dir),
        "--total-steps",
        str(args.steps),
        "--eval-split-every",
        str(args.eval_split_every),
        "--eval-every",
        str(args.eval_every),
        "--eval-save-to-disk",
        "--export-every",
        str(export_every),
        "--export-path",
        f"./runs/{experiment_name}",
    ]
    if args.with_viewer:
        cmd.append("--with-viewer")

    stream_process(cmd, run_dir / "train.log", work_root)


if __name__ == "__main__":
    main()
