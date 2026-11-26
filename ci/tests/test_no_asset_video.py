"""Video goes into recordings as ``rr.VideoStream``, never as a whole-file ``AssetVideo``.

A VideoStream is indexed per sample, so a reader can fetch one GOP; an asset is one blob that every reader loads
whole, and a ``VideoFrameReference`` pointing at it carries the same cost. ``simplecv.rerun_log_utils.log_video`` is
the one way an mp4 goes into a recording.
"""

import subprocess
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
EXCLUDED: tuple[str, ...] = (
    ":!pixi.lock",
    ":!ci/tests/test_no_asset_video.py",
    # Proposed for deletion in the architecture review; not migrated.
    ":!packages/robocap-slam",
    # Pending removal: fb14b00c's branch deletes this AssetVideo-rejecting guard and its test; drop both lines once it merges.
    ":!packages/mv-api/src/mv_api/api/catalog_prediction_layer.py",
    ":!packages/mv-api/tests/test_catalog_prediction_layer.py",
    # Pending: the same branch rewords this docstring that names AssetVideo; drop the line once it merges.
    ":!packages/mv-api/src/mv_api/api/metric_exoego_calib.py",
)
ASSET_MODE_PATTERN: str = r"""mode\s*=\s*["']asset["']"""
ASSET_MODE_ALLOWED: str = "packages/simplecv/simplecv/rerun_log_utils.py"
"""``mp4_frame_timestamps_ns`` reads only Mp4Reader's asset-mode frame index (no decode, no transcode); it logs nothing."""


def _grep(pattern: str) -> list[str]:
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["git", "grep", "-n", "-I", "-E", pattern, "--", ".", *EXCLUDED], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode in (0, 1), result.stderr
    return result.stdout.splitlines()


def test_no_asset_video_anywhere() -> None:
    hits: list[str] = _grep(r"AssetVideo|asset_video")
    assert not hits, "Log video as rr.VideoStream (simplecv.rerun_log_utils.log_video), not AssetVideo:\n" + "\n".join(hits)


def test_mp4_reader_asset_mode_only_for_the_frame_index() -> None:
    asset_mode: list[str] = _grep(ASSET_MODE_PATTERN)
    # Guards the guard: a git grep that silently matched nothing would pass every check.
    assert any(line.startswith(f"{ASSET_MODE_ALLOWED}:") for line in asset_mode)
    hits: list[str] = [line for line in asset_mode if not line.startswith(f"{ASSET_MODE_ALLOWED}:")]
    assert not hits, "Mp4Reader asset mode logs a whole-file AssetVideo; use stream mode (log_video):\n" + "\n".join(hits)
