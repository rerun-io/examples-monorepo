"""Private subprocess entry point for motion inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fdanyone.device import select_cuda_device
from fdanyone.motion.gvhmr import run_gvhmr
from fdanyone.video import load_canonical_working_clip


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text())
    device, _ = select_cuda_device(request["device"])
    clip = load_canonical_working_clip(request["working_video"], request["clip_metadata"])
    common = {
        "clip": clip,
        "working_video": request["working_video"],
        "output_dir": request["output_dir"],
        "device": device,
    }
    result = run_gvhmr(gvhmr_root=request["gvhmr_root"], **common)
    result.save(request["result_dir"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m fdanyone.motion.worker REQUEST.json")
    main(sys.argv[1])
