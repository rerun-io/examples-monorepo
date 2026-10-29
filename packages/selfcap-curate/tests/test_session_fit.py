"""Synthetic camera and time fits, independent of raw data or a catalog."""

import numpy as np
import pytest

from selfcap_curate.quest_fit import Correspondences, FitMetrics, dlt, residuals
from selfcap_curate.records import HandTrack, Lag, Links, PhoneDetections
from selfcap_curate.session_fit import EpisodeLinks, FitEpisode, fill_phone_offsets, fit_phone
from selfcap_curate.sync_policy import fit_session


def test_dlt_recovers_a_known_projective_camera() -> None:
    rng = np.random.default_rng(12)
    world = rng.normal(size=(100, 3))
    world[:, 2] += 5.0
    camera = np.array([[750.0, 0.0, 320.0, 10.0], [0.0, 730.0, 240.0, -15.0], [0.0, 0.0, 1.0, 0.2]])
    image = np.c_[world, np.ones(len(world))] @ camera.T
    points = image[:, :2] / image[:, 2:3]
    pairs = Correspondences(world, points, np.ones(len(world)))
    recovered = dlt(pairs)
    assert np.max(residuals(recovered, pairs)) < 1e-07


def synthetic_motion(tau_s: float):
    """Two independently moving non-planar hands seen by a known camera."""

    rng = np.random.default_rng(91)
    geometry = rng.normal(scale=0.1, size=(21, 3))
    geometry[:, 2] += 4.0
    times = np.arange(0.0, 6.0, 0.001)
    camera = np.array([[750.0, 0.0, 320.0, 10.0], [0.0, 730.0, 240.0, -15.0], [0.0, 0.0, 1.0, 0.2]])

    def positions(stamps, side):
        phase = stamps[:, None] * (0.6 + np.arange(21)[None, :] * 0.04) + side
        return geometry[None] + np.stack([np.sin(phase) * 0.2, np.cos(phase * 1.3) * 0.3, np.sin(phase * 0.8) * 0.15], -1)

    tracks = {
        side: HandTrack(times, positions(times, i), np.ones(len(times), dtype=bool), np.ones(len(times))) for i, side in enumerate(("left", "right"))
    }
    phone_times = np.linspace(0.5, 5.5, 30)
    xy = np.full((len(phone_times), 133, 2), np.nan)
    scores = np.zeros((len(phone_times), 133))
    for side, base in enumerate((91, 112)):
        points = positions(phone_times + tau_s, side)
        image = np.c_[points.reshape(-1, 3), np.ones(len(phone_times) * 21)] @ camera.T
        xy[:, base : base + 21] = (image[:, :2] / image[:, 2:3]).reshape(-1, 21, 2)
        scores[:, base : base + 21] = 1.0
    return (PhoneDetections(phone_times, xy, scores), tracks)


def test_known_time_lag_is_recovered_within_a_tenth_millisecond() -> None:

    detections, quest = synthetic_motion(0.04)
    fit = fit_phone({"ep001": detections}, {"ep001": quest}, np.arange(-0.1, 0.100001, 0.02))
    assert fit["ep001"].tau_ms is not None
    assert abs(fit["ep001"].tau_ms - 40.0) <= 0.1


@pytest.mark.parametrize(("tau_s", "expected_choice"), [(0.005, "same_clock"), (0.5, "own_weak_fit")])
def test_never_good_phone_chooses_weak_clock_only_with_a_three_pixel_gain(tau_s: float, expected_choice: str) -> None:

    detections, quest = synthetic_motion(tau_s)
    missing = Lag(None, None, None, 0.0)
    observation = EpisodeLinks("2025-11-24__027d8215__ep001", Links(missing, missing, missing, missing, missing), (0.0, 5.0), 0.0)
    ep = FitEpisode("session", observation, ("anchor", "weak"), {"weak": detections}, quest)
    fits = {"anchor": {"ep001": FitMetrics(0.0, 1.0, 1.0, 100)}, "weak": {"ep001": FitMetrics(tau_s * 1000.0, 20.0, 50.0, 100)}}
    fill = fill_phone_offsets([ep], fits)
    assert fill.never_fit["weak"] == expected_choice
    assert fill.phone_constants_ms["weak"] == (tau_s * 1000.0 if expected_choice == "own_weak_fit" else 0.0)


def test_a_synthetic_session_recovers_its_known_tau_across_episodes() -> None:

    detections, quest = synthetic_motion(0.04)
    measured = Links(
        Lag(-0.048, 0.9, None, 5.0), Lag(0.0, 0.9, None, 5.0), Lag(0.0, 0.9, None, 5.0), Lag(0.2, 0.9, None, 5.0), Lag(-0.248, 0.9, None, 5.0)
    )
    episodes = [
        FitEpisode(
            "027d8215-session", EpisodeLinks(f"2025-11-24__027d8215__ep{i:03d}", measured, (0.0, 5.0), 0.0), ("Phone",), {"Phone": detections}, quest
        )
        for i in (1, 2)
    ]
    records = fit_session(episodes)
    for record in records.values():
        assert record.exo_fit["Phone"].tau_phone_to_pose_ms == pytest.approx(40.0, abs=0.1)
        assert record.offsets_ms["exo/Phone"] == pytest.approx(40.0, abs=0.1)
