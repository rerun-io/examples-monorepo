"""log_relative_pred logs confidence only when the prediction has one, as grayscale + semantic mask."""

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.experimental as rrx

from monopriors.models.relative_depth import RelativeDepthPrediction
from monopriors.rr_logging_utils import log_relative_pred


def _logged_entities(tmp_path: Path, confidence: np.ndarray | None) -> set[str]:
    rrd = tmp_path / "pred.rrd"
    rec = rr.RecordingStream("test_log_relative_pred")
    rec.save(str(rrd))
    h, w = 8, 12
    depth = np.linspace(1.0, 5.0, h * w, dtype=np.float32).reshape(h, w)
    pred = RelativeDepthPrediction(disparity=1.0 / depth, depth=depth, K_33=np.eye(3, dtype=np.float32) * 10, confidence=confidence)
    with rec:
        log_relative_pred(Path("world"), pred, np.zeros((h, w, 3), dtype=np.uint8), remove_flying_pixels=False)
    rec.flush(timeout_sec=10.0)
    rec.disconnect()
    reader = rrx.RrdReader(str(rrd))
    return {str(chunk.entity_path) for store in reader.recordings() for chunk in reader.stream(store=store).to_chunks()}


def test_no_confidence_head_logs_nothing(tmp_path: Path) -> None:
    entities = _logged_entities(tmp_path, None)
    assert "/world/camera/pinhole/depth" in entities
    assert not [e for e in entities if "confidence" in e]


def test_confidence_logs_spectrum_and_mask(tmp_path: Path) -> None:
    conf = np.linspace(0.0, 1.0, 8 * 12, dtype=np.float32).reshape(8, 12)
    entities = _logged_entities(tmp_path, conf)
    assert {"/world/camera/pinhole/confidence", "/world/camera/pinhole/confidence_mask"} <= entities
