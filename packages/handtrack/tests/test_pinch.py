import numpy as np
import pytest

from handtrack.eval.scorecard import pinch_state as scorecard_pinch_state
from handtrack.pinch import OPEN_MM, PinchConfig, PinchDetector, pinch_signal, pinch_state

NAN = float("nan")


def test_enter_after_confirm_frames_and_release_with_hysteresis() -> None:
    distance = np.array([30, 9, 9, 9, 12, 15, 17, 17, 17, 9], dtype=np.float32)
    # under 10 mm for 2 frames enters on the 2nd; 12 and 15 sit in the band (no release); two frames over 16 release on the 2nd
    assert pinch_state(distance).tolist() == [False, False, True, True, True, True, True, False, False, False]


def test_untracked_frames_hold_then_release() -> None:
    distance = np.array([5, 5, NAN, NAN, NAN, 5, NAN, NAN, NAN, NAN, 5], dtype=np.float32)
    # a gap of 3 holds; the 4th untracked frame releases, and re-entering needs 2 tracked frames again
    assert pinch_state(distance).tolist() == [False, True, True, True, True, True, True, True, True, False, False]


def test_scorecard_uses_the_runtime_machine() -> None:
    rng = np.random.default_rng(0)
    distance = rng.uniform(0, 30, 400).astype(np.float32)
    distance[rng.random(400) < 0.1] = np.nan
    assert np.array_equal(scorecard_pinch_state(distance), pinch_state(distance))
    assert np.array_equal(scorecard_pinch_state(distance, 14.0, 18.0, 1, 0), pinch_state(distance, 14.0, 18.0, 1, 0))


def test_streaming_detector_reports_one_click_per_pinch() -> None:
    detector = PinchDetector(PinchConfig())
    events = [detector.step(d) for d in (30.0, 8.0, 8.0, 8.0, 20.0, 20.0, 8.0, 8.0)]
    assert [e.onset for e in events] == [False, False, True, False, False, False, False, True]
    assert [e.release for e in events] == [False, False, False, False, False, True, False, False]
    assert [e.pinched for e in events] == pinch_state(np.array([30, 8, 8, 8, 20, 20, 8, 8], dtype=np.float32)).tolist()


def test_head_gates_the_fitted_contact() -> None:
    contact = np.array([4.0, 4.0, 4.0, NAN, 25.0], dtype=np.float32)
    head = np.array([0.9, 0.2, NAN, 0.9, 0.9], dtype=np.float32)
    gated = pinch_signal(contact, head, PinchConfig(source="fit_and_head"))
    np.testing.assert_array_equal(gated, np.array([4.0, OPEN_MM, OPEN_MM, NAN, 25.0], dtype=np.float32))
    alone = pinch_signal(contact, head, PinchConfig(source="head"))
    np.testing.assert_array_equal(alone, np.array([0.0, OPEN_MM, NAN, NAN, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(pinch_signal(contact, None, PinchConfig()), contact)
    with pytest.raises(ValueError, match="pinch head"):
        pinch_signal(contact, None, PinchConfig(source="head"))


def test_streaming_step_with_a_head_matches_the_batch_signal() -> None:
    config = PinchConfig(source="fit_and_head", enter_mm=14.0, leave_mm=18.0, frames=1, head_threshold=0.3)
    contact = np.array([20, 10, 10, 10, 10, 25], dtype=np.float32)
    head = np.array([0.1, 0.1, 0.8, 0.8, 0.1, 0.8], dtype=np.float32)
    detector = PinchDetector(config)
    streamed = [detector.step(float(c), float(h)).pinched for c, h in zip(contact, head, strict=True)]
    assert streamed == pinch_state(pinch_signal(contact, head, config), 14.0, 18.0, 1, config.hold).tolist() == [False, False, True, True, False, False]
