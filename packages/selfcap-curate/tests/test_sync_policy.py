"""Every fallback flag and the additive phone-fill rule has an explicit trigger."""

from dataclasses import replace

import pytest

from selfcap_curate.quest_fit import FitMetrics
from selfcap_curate.records import Lag, Links, RepeatPattern
from selfcap_curate.session_fit import EpisodeLinks, FitEpisode, PhoneFill, fill_phone_offsets, good_fit
from selfcap_curate.sync_policy import LinkMedians, RgbSession, resolve_link_offsets, resolve_rgb, rgb_session, sync_records


def lag(value: float | None, peak: float | None = 0.9) -> Lag:
    return Lag(value, peak, None, 50.0)


def links() -> Links:
    return Links(lag(-0.1), lag(0.01), lag(0.01), lag(0.2), lag(-0.3))


def episode(ep: str = "ep001", *, phone_names: tuple[str, ...] = (), measurement: Links | None = None) -> FitEpisode:
    return FitEpisode(
        "027d8215-7ef4-418e-bc3b-fb98f3a3ae06",
        EpisodeLinks(f"2025-11-24__027d8215__{ep}", measurement or links(), (0.0, 50.0), 10.0),
        phone_names,
        {},
        {},
    )


@pytest.mark.parametrize(
    ("flag", "changes", "medians", "field", "value"),
    [
        (
            "quest_unsynced_kept_as_cut",
            {"lag_left_to_pose": lag(-0.1, 0.1), "lag_left_to_gyro": lag(0.2, 0.1), "lag_gyro_to_pose": lag(-0.3, 0.1)},
            LinkMedians(None, None, None),
            "pose_ms",
            0.0,
        ),
        ("quest_pose_via_gyro", {"lag_left_to_pose": lag(-0.15)}, LinkMedians(None, None, None), "pose_ms", 100.0),
        (
            "quest_pose_low_confidence",
            {"lag_left_to_pose": lag(-0.1, 0.45), "lag_left_to_gyro": lag(0.2, 0.1), "lag_gyro_to_pose": lag(-0.3, 0.1)},
            LinkMedians(None, None, None),
            "pose_ms",
            100.0,
        ),
        (
            "quest_pose_session_median",
            {"lag_left_to_pose": lag(None, 0.7), "lag_left_to_gyro": lag(None), "lag_gyro_to_pose": lag(None)},
            LinkMedians(88.0, None, None),
            "pose_ms",
            88.0,
        ),
        (
            "quest_pose_unsynced",
            {"lag_left_to_pose": lag(None, 0.7), "lag_left_to_gyro": lag(None), "lag_gyro_to_pose": lag(None)},
            LinkMedians(None, None, None),
            "pose_ms",
            0.0,
        ),
        ("oak_imu_via_pose", {"lag_left_to_gyro": lag(0.2, 0.1)}, LinkMedians(None, None, None), "imu_ms", -200.0),
        (
            "oak_imu_low_confidence",
            {"lag_left_to_gyro": lag(0.2, 0.1), "lag_gyro_to_pose": lag(-0.3, 0.1)},
            LinkMedians(None, None, None),
            "imu_ms",
            -200.0,
        ),
        (
            "oak_imu_session_median",
            {"lag_left_to_gyro": lag(None, None), "lag_gyro_to_pose": lag(None, None)},
            LinkMedians(None, -22.0, None),
            "imu_ms",
            -22.0,
        ),
        (
            "oak_imu_unsynced",
            {"lag_left_to_gyro": lag(None, None), "lag_gyro_to_pose": lag(None, None)},
            LinkMedians(None, None, None),
            "imu_ms",
            0.0,
        ),
        ("quest_video_low_confidence", {"lag_quest_video_to_pose": lag(0.01, 0.1)}, LinkMedians(None, None, None), "video_ms", 110.0),
        ("quest_video_session_median", {"lag_quest_video_to_pose": lag(None)}, LinkMedians(None, None, 15.0), "video_ms", 115.0),
        ("quest_video_from_direct_video_link", {"lag_left_to_quest_video": lag(-0.3, 0.95)}, LinkMedians(None, None, None), "video_ms", 300.0),
        ("quest_video_via_gyro", {"left_to_gyro": lag(0.2), "quest_video_to_gyro": lag(0.35)}, LinkMedians(None, None, None), "video_ms", 150.0),
    ],
)
def test_link_flag_marks_its_clock_fallback(flag, changes, medians, field, value) -> None:
    result = resolve_link_offsets(replace(links(), **changes), medians)
    assert flag in result.flags
    assert getattr(result, field) == pytest.approx(value)


@pytest.mark.parametrize(
    ("flag", "measurement", "model"),
    [
        ("oak_rgb_low_confidence", lag(0.06, 0.5), RgbSession(0.0, 0.3, 1, {})),
        ("oak_rgb_episode_disagrees", lag(0.06, 0.1), RgbSession(0.0, 0.3, 1, {})),
        ("oak_rgb_without_gyro", lag(0.01, 0.5), RgbSession(0.0, 0.3, 1, {})),
        ("oak_rgb_drift_default", lag(0.01, 0.5), RgbSession(0.0, 0.3, 0, {})),
    ],
)
def test_rgb_flag_exposes_its_evidence_limit(flag, measurement, model) -> None:
    result = resolve_rgb(episode(measurement=replace(links(), lag_rgb_to_left=measurement)), model)
    assert flag in result.flags


def test_confident_rgb_gyro_measurement_anchors_the_midpoint() -> None:
    result = resolve_rgb(episode(), RgbSession(0.0, 0.3, 1, {"ep001": 22.0}))
    assert result.model.how == "episode_via_gyro"
    assert result.model.a_ms + result.model.b_ms_per_s * 35.0 == pytest.approx(22.0)
    assert "oak_rgb_without_gyro" not in result.flags


def test_clipped_peak_is_missing_and_flagged_in_the_record() -> None:
    ep = episode(measurement=replace(links(), lag_left_to_gyro=lag(0.246)))
    clipped = replace(
        ep.observation.links.lag_left_to_gyro, lag_s=None, peak=None, clipped=True, clipped_at_s=ep.observation.links.lag_left_to_gyro.lag_s
    )
    ep = replace(ep, observation=replace(ep.observation, links=replace(ep.observation.links, lag_left_to_gyro=clipped)))
    assert ep.observation.links.lag_left_to_gyro.lag_s is None
    record = sync_records([ep], {}, PhoneFill({"ep001": None}, {}, {}))[ep.key]
    assert "links_clipped_at_search_edge" in record.flags


def test_failed_links_leave_offsets_unscored() -> None:
    ep = episode()
    ep = replace(ep, observation=replace(ep.observation, failed=True))
    record = sync_records([ep], {}, PhoneFill({}, {}, {}))[ep.key]
    assert record.flags == ["links_failed"]
    assert record.offsets_ms is None


def trusted(tau: float) -> FitMetrics:
    return FitMetrics(tau, 2.0, 5.0, 100, False)


def test_fill_model_recovers_missing_phone_clock_from_session_constants() -> None:
    episodes = [episode(f"ep{i:03d}", phone_names=("A", "B")) for i in (1, 2, 3)]
    fits = {"A": {"ep001": trusted(100.0), "ep002": trusted(200.0), "ep003": trusted(300.0)}, "B": {"ep001": trusted(120.0), "ep002": trusted(220.0)}}
    fill = fill_phone_offsets(episodes, fits)
    assert fill.phone_constants_ms == {"A": -10.0, "B": 10.0}
    assert fill.episode_time_ms["ep003"] == 310.0
    record = sync_records(episodes, fits, fill)[episodes[2].key]
    assert record.offsets_ms["exo/B"] == 372.0
    assert "exo_B_filled" in record.flags


@pytest.mark.parametrize(
    ("flag", "fit", "common", "constant", "choice", "expected"),
    [
        ("exo_P_filled", trusted(130.0), 100.0, 0.0, None, 152.0),
        ("exo_P_weak", FitMetrics(120.0, 20.0, 50.0, 20, False), 100.0, 20.0, "own_weak_fit", 172.0),
        ("exo_P_unverified", None, 100.0, 0.0, "same_clock", 152.0),
        ("exo_P_unsynced", None, None, 0.0, None, None),
    ],
)
def test_phone_fallback_flags_distinguish_measurement_from_inference(flag, fit, common, constant, choice, expected) -> None:
    ep = episode(phone_names=("P",))
    fit_map = {"P": {"ep001": fit} if fit is not None else {}}
    fill = PhoneFill({"ep001": common}, {"P": constant}, {"P": choice} if choice is not None else {})
    record = sync_records([ep], fit_map, fill)[ep.key]
    assert flag in record.flags
    assert record.offsets_ms["exo/P"] == expected


@pytest.mark.parametrize(
    "bad",
    [
        FitMetrics(None, 2.0, 5.0, 100),
        FitMetrics(0.0, 2.0, 5.0, 59),
        FitMetrics(0.0, 14.1, 5.0, 100),
        FitMetrics(0.0, 2.0, 40.1, 100),
        FitMetrics(0.0, 2.0, 5.0, 100, True),
    ],
)
def test_bad_fit_cannot_anchor_other_phones(bad: FitMetrics) -> None:
    assert not good_fit(bad)


def test_rgb_drift_uses_relative_pattern_rate_and_rejects_low_agreement() -> None:

    left = RepeatPattern(10, 0.0, 0.99, 500, 1.0, 0.1, 1200)
    rgb = replace(left, ppm_vs_24=310)
    good = episode(measurement=replace(links(), oak_left_pattern=left, oak_rgb_pattern=rgb))
    bad = episode("ep002", measurement=replace(links(), oak_left_pattern=left, oak_rgb_pattern=replace(rgb, ppm_vs_24=2000, agreement=0.5)))
    model = rgb_session([good, bad])
    assert model.b_ms_per_s == 0.3
    assert model.ppm_count == 1


def test_rgb_flags_unknown_session_origin() -> None:
    ep = episode(measurement=replace(links(), lag_rgb_to_left=lag(None)))
    ep = replace(ep, observation=replace(ep.observation, t_start_session_s=None))
    result = resolve_rgb(ep, RgbSession(0.0, 0.3, 1, {}))
    assert "oak_rgb_session_origin_unknown" in result.flags
