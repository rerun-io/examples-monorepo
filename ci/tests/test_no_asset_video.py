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
)
ASSET_MODE_PATTERN: str = r"""mode\s*=\s*["']asset["']"""


def _grep(pattern: str) -> list[str]:
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["git", "grep", "-n", "-I", "-E", pattern, "--", ".", *EXCLUDED], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode in (0, 1), result.stderr
    return result.stdout.splitlines()


def test_no_asset_video_anywhere() -> None:
    hits: list[str] = _grep(r"AssetVideo|asset_video")
    assert not hits, "Log video as rr.VideoStream (simplecv.rerun_log_utils.log_video), not AssetVideo:\n" + "\n".join(hits)


def test_no_mp4_reader_asset_mode() -> None:
    hits: list[str] = _grep(ASSET_MODE_PATTERN)
    assert not hits, "Mp4Reader asset mode logs a whole-file AssetVideo; use stream mode (log_video), or Mp4Reader.info() for frame times:\n" + "\n".join(hits)
