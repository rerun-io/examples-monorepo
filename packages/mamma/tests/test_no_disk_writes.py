"""Goal clause 4: the streaming loop performs no intermediate disk writes.

Two layers:
1. Static audit — no file-write calls in the hot-loop modules.
2. Runtime check — a short end-to-end run (when local data exists) creates no
   files anywhere under the package tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_PKG: Path = Path(__file__).parent.parent
_SRC: Path = _PKG / "src" / "mamma"
_HOT_LOOP_MODULES: tuple[str, ...] = ("engine", "tracking", "landmarks", "fitting", "viz", "eval")
_WRITE_PATTERNS: re.Pattern = re.compile(
    r"""open\([^)]*['"][wax]b?['"]|\.write_text\(|\.write_bytes\(|np\.save|np\.savez|torch\.save|\.to_csv\(|cv2\.imwrite|imageio\.|mkdir\(""",
)

_DATA: Path = _PKG / "data"
needs_data = pytest.mark.skipif(
    not (_DATA / "inputs/indoors/crossing_arms/meta/global.npz").exists()
    or not (_DATA / "weights/ma_2d/mamma_mask_full_cvpr.safetensors").exists(),
    reason="local data/weights not downloaded",
)


def test_hot_loop_modules_have_no_write_calls() -> None:
    offenders: list[str] = []
    for module in _HOT_LOOP_MODULES:
        for py in (_SRC / module).rglob("*.py"):
            for lineno, line in enumerate(py.read_text().splitlines(), start=1):
                if _WRITE_PATTERNS.search(line) and "rb" not in line:
                    offenders.append(f"{py.relative_to(_PKG)}:{lineno}: {line.strip()}")
    assert not offenders, "file-write calls found in streaming modules:\n" + "\n".join(offenders)


@needs_data
def test_streaming_run_creates_no_files() -> None:
    import rerun as rr
    from simplecv.video_io import TorchCodecMultiVideoReader

    from mamma.datasets.mamma_npz import load_mamma_sequence
    from mamma.engine.pipeline import StreamingPipeline
    from mamma.fitting.stage import FittingStage
    from mamma.fitting.window_fitter import FitterConfig
    from mamma.landmarks.estimator import LandmarkEstimator
    from mamma.tracking.tracker import MultiViewTracker, TrackerConfig
    from mamma.viz.stream_logger import StreamLogger

    rr.init("no-disk-writes-test", strict=True)
    memory_sink = rr.memory_recording()  # noqa: F841 — keeps logs in RAM

    before: set[Path] = {p for p in _PKG.rglob("*") if p.is_file()}

    sequence = load_mamma_sequence(_DATA / "inputs/indoors/crossing_arms")
    resize_hw: tuple[int, int] = (720, 1280)
    scaled = [c.scaled_to(height=resize_hw[0], width=resize_hw[1]) for c in sequence.cameras]
    tracker_cfg = TrackerConfig(
        sam2_checkpoint=_DATA / "weights/efficienttam/efficienttam_ti.pt",
        yolo_checkpoint=_DATA / "weights/yolo/yolo12x.pt",
        expected_subjects=1,
    )
    fitter_cfg = FitterConfig(
        smplx_model_dir=_DATA / "body_models",
        downsampled_verts_pkl=_DATA / "body_models/downsampled_verts/verts_512.pkl",
        window_size=4,
        bootstrap_iters=10,
        tick_iters=2,
    )
    pipeline = StreamingPipeline(
        sequence,
        TorchCodecMultiVideoReader(list(sequence.video_paths), device="cuda", resize_hw=resize_hw),
        StreamLogger(sequence, resize_hw=resize_hw),
        tracker=MultiViewTracker(scaled, tracker_cfg),
        landmarks=LandmarkEstimator(_DATA / "weights/ma_2d/mamma_mask_full_cvpr.safetensors"),
        fitting=FittingStage(scaled, fitter_cfg),
    )
    pipeline.run(max_frames=8, start_frame=60)

    after: set[Path] = {p for p in _PKG.rglob("*") if p.is_file()}
    new_files: set[Path] = after - before
    assert not new_files, f"streaming run created files: {sorted(new_files)[:10]}"
