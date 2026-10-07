"""Saved child conversation rows must appear in the headless viewer's pixels."""

import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from PIL import Image
from rerun.experimental import ViewerClient

from agent_traces.events import Prompt, Session, TimedRecord, ToolCall, ToolResult, Usage, UsageSample
from agent_traces.rerun_log import write_session_rrd


def capture_recording(recording: Path, screenshot: Path, port: int, binary: Path) -> None:
    """Run the viewer in a bounded subprocess so a failed render cannot wedge pytest."""
    with ViewerClient.spawn(headless=True, port=port, hide_welcome_screen=True, executable_path=str(binary)) as viewer:
        rr.init("child_pixel_check", default_enabled=True, strict=True)
        rr.connect_grpc(url=viewer.url)
        rr.log_file_from_path(recording)
        stream = rr.get_global_data_recording()
        assert stream is not None
        try:
            stream.flush(timeout_sec=30.0)
            time.sleep(3.0)
            viewer.save_screenshot(str(screenshot))
        finally:
            stream.disconnect()


@pytest.mark.golden
def test_child_conversation_rows_render(tmp_path: Path, rerun_binary: Path, free_port: int) -> None:
    """Child prefixes render in Conversation and Tools; plots label separate main and children lines."""
    version = subprocess.run([str(rerun_binary), "--version"], capture_output=True, text=True, check=True)
    assert rr.__version__ in version.stdout, version.stdout
    session: Session = Session("child-pixels", "synthetic", tmp_path / "source.jsonl", [
        TimedRecord(UsageSample(Usage(input_tokens=10)), 1_000_000_000, 0),
        TimedRecord(UsageSample(Usage(input_tokens=20)), 2_000_000_000, 1),
        TimedRecord(ToolResult("Read", "main", "main result", "{}", "file_read", 4.0), 1_000_000_000, 2),
        TimedRecord(ToolResult("Read", "main2", "main result", "{}", "file_read", 8.0), 2_000_000_000, 3),
    ], {
        "child": [TimedRecord(Prompt("CHILD ROW: visible conversation text from a nested agent"), 1_000_000_000, 0),
                  TimedRecord(ToolCall("Read", "child", "{}", "file_read"), 1_000_000_000, 1),
                  TimedRecord(ToolResult("Read", "child", "child result", "{}", "file_read", 2.0), 1_000_000_000, 2),
                  TimedRecord(UsageSample(Usage(input_tokens=5)), 1_000_000_000, 3),
                  TimedRecord(ToolResult("Read", "child2", "child result", "{}", "file_read", 3.0), 2_000_000_000, 4),
                  TimedRecord(UsageSample(Usage(input_tokens=6)), 2_000_000_000, 5)]
    }, Counter())
    recording: Path = write_session_rrd(session, tmp_path / "child.rrd", host="synthetic").path
    screenshot: Path = tmp_path / "child.png"
    environment: dict[str, str] = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "config")}
    subprocess.run([str(rerun_binary), "analytics", "disable"], env=environment, capture_output=True, check=True, timeout=15)
    capture = subprocess.run([sys.executable, str(Path(__file__)), str(recording), str(screenshot), str(free_port), str(rerun_binary)],
                             env=environment, capture_output=True, text=True, timeout=60)
    diagnostics: str = capture.stdout + capture.stderr
    if "No graphics adapter was found" in diagnostics:
        pytest.skip("No graphics adapter available for the headless Rerun viewer")
    assert capture.returncode == 0, diagnostics
    with Image.open(screenshot) as rendered:
        assert rendered.size == (1920, 1080)
        # Reviewed glyph masks from the synthetic fixture, fixed at 1920x1080.
        # Check the actual [a0] glyphs in both text views, not just arbitrary ink.
        fixtures: Path = Path(__file__).parent / "golden"
        regions: list[tuple[str, tuple[int, int, int, int]]] = [
            ("child-prefix", (362, 97, 387, 111)),
            ("child-prefix", (362, 597, 387, 611)),
            ("token-legend", (1097, 368, 1232, 407)),
            ("elapsed-legend", (1097, 768, 1204, 807)),
        ]
        for name, region in regions:
            actual = np.asarray(rendered.convert("RGB").crop(region)).max(axis=2) > 70
            with Image.open(fixtures / f"{name}.png") as reference:
                expected = np.asarray(reference)
            assert float(np.mean(actual == expected)) > 0.97, f"Missing {name} glyphs at {region}: {screenshot}"



if __name__ == "__main__":
    capture_recording(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
