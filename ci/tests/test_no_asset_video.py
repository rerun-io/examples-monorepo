"""Video goes into recordings as ``rr.VideoStream``, never as a whole-file ``AssetVideo``.

A VideoStream is indexed per sample, so a reader can fetch one GOP; an asset is one blob that every reader loads
whole, and a ``VideoFrameReference`` pointing at it carries the same cost. ``simplecv.rerun_log_utils.log_video`` is
the one way an mp4 goes into a recording.
"""

import ast
import subprocess
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
EXCLUDED: tuple[str, ...] = (
    ":!pixi.lock",
    ":!ci/tests/test_no_asset_video.py",
)
ASSET_MODE_PATTERN: str = r"""mode\s*=\s*["']asset["']"""
ASSET_MODE_ALLOWED: tuple[str, str] = ("packages/simplecv/simplecv/rerun_log_utils.py", "mp4_frame_timestamps_ns")
"""The one (file, function) that may call Mp4Reader in asset mode: it reads only the frame index and logs nothing."""


def _grep(pattern: str) -> list[str]:
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["git", "grep", "-n", "-I", "-E", pattern, "--", ".", *EXCLUDED], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode in (0, 1), result.stderr
    return result.stdout.splitlines()


def test_no_asset_video_anywhere() -> None:
    hits: list[str] = _grep(r"AssetVideo|asset_video")
    assert not hits, "Log video as rr.VideoStream (simplecv.rerun_log_utils.log_video), not AssetVideo:\n" + "\n".join(hits)


def _asset_mode_calls(source: str, path: str) -> list[str]:
    """``path:line in function`` for every call that passes ``mode="asset"``."""
    calls: list[str] = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call) and any(
                keyword.arg == "mode" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "asset" for keyword in child.keywords
            ):
                calls.append(f"{path}:{child.lineno} in {function}")
            visit(child, function)

    visit(ast.parse(source), "<module>")
    return calls


def test_mp4_reader_asset_mode_only_for_the_frame_index() -> None:
    files: set[str] = {line.split(":", 1)[0] for line in _grep(ASSET_MODE_PATTERN)}
    assert ASSET_MODE_ALLOWED[0] in files, "the frame-index helper's asset-mode call is gone; update ASSET_MODE_ALLOWED"
    calls: list[str] = [
        call
        for path in sorted(files)
        for call in (_asset_mode_calls((REPO_ROOT / path).read_text(), path) if path.endswith(".py") else [f"{path} (not Python)"])
    ]
    allowed: str = f"{ASSET_MODE_ALLOWED[0]}:"
    hits: list[str] = [call for call in calls if not (call.startswith(allowed) and call.endswith(f" in {ASSET_MODE_ALLOWED[1]}"))]
    assert not hits, "Mp4Reader asset mode logs a whole-file AssetVideo; use stream mode (log_video):\n" + "\n".join(hits)


def test_asset_mode_check_names_the_enclosing_function() -> None:
    source: str = 'def mp4_frame_timestamps_ns():\n    Mp4Reader(p, mode="asset")\n\n\ndef log_video():\n    Mp4Reader(p, mode="asset")\n'
    assert _asset_mode_calls(source, "f.py") == ["f.py:2 in mp4_frame_timestamps_ns", "f.py:6 in log_video"]
