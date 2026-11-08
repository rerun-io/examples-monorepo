"""The chase eye rides the body: placed behind and above it along its own axes, aimed ahead."""

import pytest

from simplecv.rerun_eye_controls import follow_eye_controls


def _xyz(batch: object) -> list[float]:
    return batch.as_arrow_array().to_pylist()[0]  # pyrefly: ignore


def test_follow_eye_sits_behind_and_above_along_the_body_axes() -> None:
    # An RDF camera: forward +z, up -y.
    eye = follow_eye_controls((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), back_m=1.5, up_m=0.6, ahead_m=1.5)
    assert _xyz(eye.position) == pytest.approx([0.0, -0.6, -1.5])
    assert _xyz(eye.look_target) == pytest.approx([0.0, 0.0, 1.5])
    assert _xyz(eye.eye_up) == pytest.approx([0.0, -1.0, 0.0])
