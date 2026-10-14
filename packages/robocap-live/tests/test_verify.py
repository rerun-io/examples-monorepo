"""The graph-tier score on golden-shaped tables (unit lane: the two golden reads are faked, no catalog server)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from dagkit.verify import Case, load_card

from robocap_live import verify


def golden(keynet_rows: list[bool]) -> pa.Table:
    """What ``_read`` returns: one camera pane's KeyNet points on the framesets where ``keynet_rows`` is true, one hand's world
    landmarks on all three (every point at the origin, so matched rows score 0)."""
    points2d, points3d = np.zeros((21, 2)).tolist(), np.zeros((21, 3)).tolist()
    return pa.table({
        "video_time": pa.array([1, 2, 3], type=pa.int64()),
        f"{verify.KEYNET[0]}:Points2D:positions": pa.array([points2d if row else None for row in keynet_rows], type=pa.list_(pa.list_(pa.float32(), 2))),
        f"{verify.LANDMARKS[0]}:Points3D:positions": pa.array([points3d] * 3, type=pa.list_(pa.list_(pa.float32(), 3))),
    })  # fmt: skip


def test_keynet_rows_lost_on_a_camera_pane_fail_a_band_although_the_matched_rows_score_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pre: Path = tmp_path / "short.pre.rrd"
    pre.touch()
    reference, post = golden([True, True, True]), golden([True, False, False])
    monkeypatch.setattr(verify, "_read", lambda path: (reference if path == pre else post, {}))
    scores: dict[str, float] = verify.score(Case("short", "short", "pre-port"), tmp_path / "short.post.rrd")
    assert scores["kp2d_px_mean"] == scores["landmarks_mm_mean"] == scores["reported_delta_pct"] == 0.0
    assert scores["kp2d_delta_pct"] == pytest.approx(200.0 / 3)  # 2 of the reference's 3 pane rows are gone
    block = next(b for b in load_card(Path(verify.PACKAGE_DIR) / "VERIFY.toml").graph if b.name == "hands")
    band = next(m for m in block.metrics if m.name == "kp2d_delta_pct")
    assert band.better == "lower" and scores["kp2d_delta_pct"] > band.floor
    assert all("kp2d_delta_pct" in case.noise for case in block.cases)


@pytest.mark.parametrize("suffix", ["partial", "rrd"])
def test_pre_scores_itself_with_one_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str) -> None:
    pre: Path = tmp_path / f"short.pre.{suffix}"
    pre.touch()
    reads: list[Path] = []

    def read(path: Path) -> tuple[pa.Table, dict[str, float]]:
        reads.append(path)
        return golden([True, True, True]), {}

    monkeypatch.setattr(verify, "_read", read)
    scores: dict[str, float] = verify.score(Case("short", "short", "pre-port"), pre)
    assert reads == [pre]
    assert scores["kp2d_px_mean"] == scores["landmarks_mm_mean"] == scores["reported_delta_pct"] == scores["kp2d_delta_pct"] == 0.0
