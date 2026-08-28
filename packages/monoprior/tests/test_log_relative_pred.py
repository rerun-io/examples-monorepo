"""log_relative_pred logs confidence only when the prediction has one: grayscale spectrum + static-annotated semantic mask."""

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.experimental as rrx

from monopriors.models.relative_depth import RelativeDepthPrediction
from monopriors.rr_logging_utils import log_relative_pred

H, W = 8, 12


def _logged_chunks(tmp_path: Path, confidence: np.ndarray | None) -> dict[str, list[tuple[set[str], bool]]]:
    """entity path -> [(column names, is_static)] for everything log_relative_pred wrote."""
    rrd = tmp_path / "pred.rrd"
    rec = rr.RecordingStream("test_log_relative_pred")
    rec.save(str(rrd))
    depth = np.linspace(1.0, 5.0, H * W, dtype=np.float32).reshape(H, W)
    pred = RelativeDepthPrediction(disparity=1.0 / depth, depth=depth, K_33=np.eye(3, dtype=np.float32) * 10, confidence=confidence)
    with rec:
        log_relative_pred(Path("world"), pred, np.zeros((H, W, 3), dtype=np.uint8), remove_flying_pixels=False)
    reader = rrx.RrdReader(str(rrd))
    chunks: dict[str, list[tuple[set[str], bool]]] = {}
    for store in reader.recordings():
        for chunk in reader.stream(store=store).to_chunks():
            chunks.setdefault(str(chunk.entity_path), []).append(({f.name for f in chunk.to_record_batch().schema}, chunk.is_static))
    return chunks


def test_no_confidence_head_logs_nothing(tmp_path: Path) -> None:
    chunks = _logged_chunks(tmp_path, None)
    assert "/world/camera/pinhole/depth" in chunks
    assert not [e for e in chunks if "confidence" in e]


def test_confidence_logs_grayscale_spectrum_and_semantic_mask(tmp_path: Path) -> None:
    conf = np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape(H, W)
    chunks = _logged_chunks(tmp_path, conf)
    spectrum = chunks["/world/camera/pinhole/confidence"]
    assert any("Image:buffer" in cols and not static for cols, static in spectrum)
    mask = chunks["/world/camera/pinhole/confidence_mask"]
    assert any("SegmentationImage:buffer" in cols and not static for cols, static in mask)
    assert any("AnnotationContext:context" in cols and static for cols, static in mask)
