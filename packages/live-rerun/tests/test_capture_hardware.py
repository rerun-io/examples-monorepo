"""Hardware-gated end-to-end capture test (requires an attached OAK device).

Skipped automatically when no OAK is present, so it's a no-op in CI. Run it
explicitly with a camera attached via ``pytest -m hardware``.
"""

from __future__ import annotations

import pytest

depthai = pytest.importorskip("depthai")

pytestmark = pytest.mark.hardware


def _no_device() -> bool:
    return len(depthai.Device.getAllAvailableDevices()) == 0


@pytest.mark.skipif(_no_device(), reason="no OAK device attached")
def test_capture_streams_encoded_keyframes() -> None:
    from live_rerun.sources.depthai import DepthAiConfig, OakSource

    counts: dict[str, int] = {"rgb": 0, "left": 0, "right": 0}
    keyframes: int = 0
    with OakSource(DepthAiConfig(usb2=True)) as source:
        assert len(source.calibrations) == 3
        for frame in source.frames():
            counts[frame.label] += 1
            assert len(frame.sample) > 0
            keyframes += int(frame.is_keyframe)
            if all(count >= 5 for count in counts.values()):
                break

    assert keyframes >= 1, "expected at least one keyframe (IDR) across the captured frames"
