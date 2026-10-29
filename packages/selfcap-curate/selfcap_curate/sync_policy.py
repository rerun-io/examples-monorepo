"""v2 clock-selection, RGB drift and phone-fill rules, without file or catalog IO."""

import math
from dataclasses import dataclass

import numpy as np

from selfcap_curate.records import HAND_LATENCY_MS, Confidence, ExoFit, Lag, Links, PhoneDetections, RgbModel, TimeSync
from selfcap_curate.session_fit import FitEpisode, PhoneFill, PhoneFit, coarse_grid, fill_phone_offsets, fit_phone, good_fit

VERSION: str = "selfcap-sync v2 (2026-10-13)"
FIT_VERSION: str = "selfcap-curate-cpu-v1"
RGB_B_DEFAULT: float = 0.30
"""Corpus-median ms/s drift; guards sessions without a trustworthy repeat pattern."""
RGB_FORMULA: str = "offset_ms(t) = a_ms + b_ms_per_s * (t_start_session_s + t), t = rgb PTS in s"


def lag_ms(lag: Lag | None, threshold: float = 0.5) -> float | None:
    """Only finite, interior lags with sufficient rotation correlation may anchor clocks."""
    if (
        lag is not None
        and lag.lag_s is not None
        and math.isfinite(lag.lag_s)
        and lag.peak is not None
        and math.isfinite(lag.peak)
        and lag.peak >= threshold
        and not lag.clipped
    ):
        return lag.lag_s * 1000.0
    return None


@dataclass(frozen=True, slots=True)
class LinkMedians:
    """Trusted session medians used only when an episode lacks a usable link."""

    pose_ms: float | None
    """Quest pose offset."""
    imu_ms: float | None
    """OAK IMU offset."""
    video_to_pose_ms: float | None
    """Quest video-to-pose lag."""


@dataclass(frozen=True, slots=True)
class LinkOffsets:
    """Resolved episode clocks and their selection diagnostics."""

    pose_ms: float
    """Quest pose offset to OAK left."""
    imu_ms: float
    """OAK IMU offset to OAK left."""
    video_ms: float
    """Quest video offset to OAK left."""
    closure_ms: float | None
    """Gyro-chain/direct-pose closure error."""
    flags: tuple[str, ...]
    """The selected fallback paths."""


def resolve_link_offsets(links: Links, medians: LinkMedians) -> LinkOffsets:
    """Resolve legacy link fallbacks, retaining every v2 confidence flag.

    quest_unsynced_kept_as_cut guards non-co-moving devices (all peaks <0.4).
    quest_pose_via_gyro guards >15 ms disagreement with a confident gyro chain.
    quest_pose_low_confidence preserves a finite weak direct measurement.
    quest_pose_session_median guards a missing episode link with session evidence;
    quest_pose_unsynced marks the final zero-offset fallback, never a measurement.
    oak_imu_via_pose uses a confident gyro/pose link when direct gyro evidence is weak;
    oak_imu_low_confidence retains a finite weak gyro lag; oak_imu_session_median
    uses session evidence; oak_imu_unsynced marks the zero fallback.
    quest_video_low_confidence marks a finite weak video/pose measurement;
    quest_video_session_median guards its absence. quest_video_from_direct_video_link
    trusts a ≥0.8 direct-video peak when the pose-chain video differs by >15 ms.
    quest_video_via_gyro uses the new independent gyro link, guarding video frame-grid bias.
    """
    lp, lg, gp, qv = links.lag_left_to_pose, links.lag_left_to_gyro, links.lag_gyro_to_pose, links.lag_quest_video_to_pose
    closure: float | None = None if lp.lag_s is None or lg.lag_s is None or gp.lag_s is None else (lg.lag_s + gp.lag_s - lp.lag_s) * 1000.0
    direct: Lag | None = links.lag_left_to_quest_video
    peaks = [
        -1.0,
        *(
            value
            for value in (
                lp.peak,
                min(lg.peak, gp.peak) if lg.peak is not None and gp.peak is not None else None,
                direct.peak if direct is not None else None,
            )
            if value is not None
        ),
    ]
    flags: list[str] = []
    if max(peaks) < 0.4:
        pose: float = 0.0
        flags.append("quest_unsynced_kept_as_cut")
    elif (lp_ms := lag_ms(lp)) is not None and (closure is None or abs(closure) <= 15.0):
        pose = -lp_ms
    elif lag_ms(lg) is not None and lag_ms(gp) is not None:
        assert lg.lag_s is not None and gp.lag_s is not None
        pose = -(lg.lag_s + gp.lag_s) * 1000.0
        flags.append("quest_pose_via_gyro")
    elif lp.lag_s is not None:
        pose = -lp.lag_s * 1000.0
        flags.append("quest_pose_low_confidence")
    elif medians.pose_ms is not None:
        pose = medians.pose_ms
        flags.append("quest_pose_session_median")
    else:
        pose = 0.0
        flags.append("quest_pose_unsynced")
    if (lg_ms := lag_ms(lg)) is not None:
        imu: float = -lg_ms
    elif gp.lag_s is not None and gp.peak is not None and gp.peak >= 0.5:
        imu = gp.lag_s * 1000.0 + pose
        flags.append("oak_imu_via_pose")
    elif lg.lag_s is not None:
        imu = -lg.lag_s * 1000.0
        flags.append("oak_imu_low_confidence")
    elif medians.imu_ms is not None:
        imu = medians.imu_ms
        flags.append("oak_imu_session_median")
    else:
        imu = 0.0
        flags.append("oak_imu_unsynced")
    if "quest_unsynced_kept_as_cut" in flags:
        video: float = 0.0
    elif qv.lag_s is not None:
        video = qv.lag_s * 1000.0 + pose
        if lag_ms(qv) is None:
            flags.append("quest_video_low_confidence")
    else:
        video = (medians.video_to_pose_ms or 0.0) + pose
        flags.append("quest_video_session_median")
    if "quest_unsynced_kept_as_cut" not in flags:
        if (direct_ms := lag_ms(direct, 0.8)) is not None and abs(-direct_ms - video) > 15.0:
            video = -direct_ms
            flags.append("quest_video_from_direct_video_link")
        left, quest = links.left_to_gyro, links.quest_video_to_gyro
        if lag_ms(left) is not None and lag_ms(quest) is not None:
            assert left is not None and quest is not None and left.lag_s is not None and quest.lag_s is not None
            video = (quest.lag_s - left.lag_s) * 1000.0
            flags.append("quest_video_via_gyro")
    return LinkOffsets(pose, imu, video, closure, tuple(flags))


@dataclass(frozen=True, slots=True)
class RgbSession:
    """Session RGB drift prior and trusted per-episode gyro measurements."""

    a_ms: float
    """Session intercept."""
    b_ms_per_s: float
    """Median relative repeat-rate drift, or corpus fallback."""
    ppm_count: int
    """Number of trustworthy repeat-pattern pairs."""
    gyro_rgb_ms: dict[str, float]
    """RGB-to-left offsets from two independent gyro measurements."""


def rgb_session(episodes: list[FitEpisode]) -> RgbSession:
    """Require both pattern agreements ≥0.95 to guard noisy repeat-rate estimates.

    Prefer fine gyro RGB links with both peaks ≥0.7 and no edge hit, guarding
    the frame-grid alias that can displace direct video/video peaks by 1–2 frames.
    """
    ppm: list[int] = []
    gyro: dict[str, float] = {}
    for episode in episodes:
        if episode.observation.failed:
            continue
        links = episode.observation.links
        left_pattern, rgb_pattern = links.oak_left_pattern, links.oak_rgb_pattern
        if left_pattern is not None and rgb_pattern is not None and min(left_pattern.agreement, rgb_pattern.agreement) >= 0.95:
            ppm.append(rgb_pattern.ppm_vs_24 - left_pattern.ppm_vs_24)
        left, rgb = links.left_to_gyro, links.rgb_to_gyro
        if lag_ms(left, 0.7) is not None and lag_ms(rgb, 0.7) is not None:
            assert left is not None and rgb is not None and left.lag_s is not None and rgb.lag_s is not None
            gyro[episode.episode] = (rgb.lag_s - left.lag_s) * 1000.0
    slope: float = float(np.median(ppm)) * 1e-3 if ppm else RGB_B_DEFAULT
    intercepts: list[float] = []
    for episode in episodes:
        source = episode.observation
        if source.failed or source.t_start_session_s is None:
            continue
        lag = source.links.lag_rgb_to_left
        if episode.episode in gyro:
            value = gyro[episode.episode]
        elif (value_ms := lag_ms(lag, 0.4)) is not None:
            value = value_ms
        else:
            continue
        intercepts.append(value - slope * (source.t_start_session_s + sum(source.window_s) / 2.0))
    return RgbSession(float(np.median(intercepts)) if intercepts else 0.0, slope, len(ppm), gyro)


@dataclass(frozen=True, slots=True)
class RgbChoice:
    """Per-episode drift model with its diagnostic disagreement."""

    model: RgbModel
    """Selected model."""
    offset_at_start_ms: float
    """RGB offset at episode start, before record rounding."""
    disagreement_ms: float | None
    """Episode measurement minus the unadjusted session model at the midpoint."""
    flags: tuple[str, ...]
    """RGB confidence and fallback evidence."""


def resolve_rgb(episode: FitEpisode, session: RgbSession) -> RgbChoice:
    """Prefer gyro, then ≥0.85 direct measurement, then the session model.

    oak_rgb_low_confidence marks a ≥0.4 peak displaced >30 ms from the model;
    this guards the observed 36–84 ms session-model misfits without hiding weak
    evidence. oak_rgb_episode_disagrees marks a >25 ms weak disagreement.
    oak_rgb_without_gyro exposes possible video-grid bias. oak_rgb_drift_default
    marks the 0.30 ms/s fallback when no repeat pattern constrains drift.
    """
    source = episode.observation
    lag = source.links.lag_rgb_to_left
    start: float = source.t_start_session_s or 0.0
    midpoint: float = start + sum(source.window_s) / 2.0
    model_mid: float = session.a_ms + session.b_ms_per_s * midpoint
    measured: float | None = None if lag.lag_s is None else lag.lag_s * 1000.0
    intercept: float = session.a_ms
    how: str = "session_model"
    flags: list[str] = []
    if episode.episode in session.gyro_rgb_ms:
        measured = session.gyro_rgb_ms[episode.episode]
        intercept, how = measured - session.b_ms_per_s * midpoint, "episode_via_gyro"
    elif measured is not None and lag.peak is not None and lag.peak >= 0.85:
        intercept, how = measured - session.b_ms_per_s * midpoint, "episode_measurement"
    elif measured is not None and lag.peak is not None and lag.peak >= 0.4 and abs(measured - model_mid) > 30.0:
        intercept, how = measured - session.b_ms_per_s * midpoint, "episode_measurement_model_misfit"
        flags.append("oak_rgb_low_confidence")
    elif measured is not None and abs(measured - model_mid) > 25.0:
        flags.append("oak_rgb_episode_disagrees")
    if source.t_start_session_s is None:
        flags.append("oak_rgb_session_origin_unknown")
    if how != "episode_via_gyro":
        flags.append("oak_rgb_without_gyro")
    if session.ppm_count == 0:
        flags.append("oak_rgb_drift_default")
    model: RgbModel = RgbModel(round(intercept, 2), round(session.b_ms_per_s, 4), start, how, RGB_FORMULA)
    return RgbChoice(model, intercept + session.b_ms_per_s * start, None if measured is None else measured - model_mid, tuple(flags))


def sync_records(episodes: list[FitEpisode], fits: dict[str, PhoneFit], fill: PhoneFill) -> dict[str, TimeSync]:
    """Assemble strict sync records and expose all fill/failure choices.

    links_failed makes all offsets unscored. A good phone estimate displaced
    >12 ms from T_episode+c_phone is filled too, guarding a single bad minimum.
    exo_<phone>_weak marks the ≥3 px own-weak-clock choice; _unverified marks a
    never-good phone on the common clock; _filled marks a missing/outlying fit;
    _unsynced leaves the offset null when no episode clock can be inferred.
    """
    models: RgbSession = rgb_session(episodes)
    good = [episode.observation.links for episode in episodes if not episode.observation.failed]
    poses = [-ms for links in good if (ms := lag_ms(links.lag_left_to_pose)) is not None]
    imus = [-ms for links in good if (ms := lag_ms(links.lag_left_to_gyro)) is not None]
    video_lags = [ms for links in good if (ms := lag_ms(links.lag_quest_video_to_pose)) is not None]
    medians: LinkMedians = LinkMedians(
        float(np.median(poses)) if poses else None, float(np.median(imus)) if imus else None, float(np.median(video_lags)) if video_lags else None
    )
    result: dict[str, TimeSync] = {}
    for episode in episodes:
        source = episode.observation
        links = source.links
        clocks: LinkOffsets = resolve_link_offsets(links, medians)
        rgb: RgbChoice = resolve_rgb(episode, models)
        flags: list[str] = list(clocks.flags) + list(rgb.flags)
        if any(
            lag is not None and lag.clipped
            for lag in (
                links.lag_left_to_pose,
                links.lag_rgb_to_left,
                links.lag_quest_video_to_pose,
                links.lag_left_to_gyro,
                links.lag_gyro_to_pose,
                links.lag_left_to_quest_video,
            )
        ):
            flags.insert(0, "links_clipped_at_search_edge")
        offsets: dict[str, float | None] = {
            "oak_left": 0.0,
            "oak_right": 0.0,
            "oak_imu": round(clocks.imu_ms, 1),
            "quest_pose": round(clocks.pose_ms, 1),
            "quest_video": round(clocks.video_ms, 1),
            "oak_rgb_at_start": round(rgb.offset_at_start_ms, 1),
        }
        exo: dict[str, ExoFit] = {}
        common: float | None = fill.episode_time_ms.get(episode.episode)
        for phone in episode.phones:
            fit = fits.get(phone, {}).get(episode.episode)
            constant = fill.phone_constants_ms.get(phone)
            choice = fill.never_fit.get(phone)
            if (
                fit is not None
                and good_fit(fit)
                and fit.tau_ms is not None
                and (common is None or (constant is not None and abs(fit.tau_ms - common - constant) <= 12.0))
            ):
                tau, how = fit.tau_ms, "measured"
            elif common is not None and constant is not None:
                tau = common + constant
                how, flag = {
                    "own_weak_fit": ("own_weak_fit", "weak"),
                    "same_clock": ("same_clock_as_other_phones", "unverified"),
                    None: ("filled_from_other_phones", "filled"),
                }[choice]
                flags.append(f"exo_{phone}_{flag}")
            else:
                tau, how = None, "unsynced"
                flags.append(f"exo_{phone}_unsynced")
            offsets[f"exo/{phone}"] = None if tau is None else round(tau - HAND_LATENCY_MS + clocks.pose_ms, 1)
            exo[phone] = ExoFit(
                None if tau is None else round(tau, 1),
                how,
                None if fit is None else fit.tau_ms,
                None if fit is None else fit.cmin_px,
                None if fit is None else fit.basin_halfwidth_ms,
                None if fit is None else fit.points,
                None if fit is None else fit.edge,
            )
        confidence: Confidence = Confidence(
            None if clocks.closure_ms is None else round(clocks.closure_ms, 1),
            links.lag_left_to_pose.peak,
            links.lag_left_to_gyro.peak,
            links.lag_quest_video_to_pose.peak,
            links.lag_rgb_to_left.peak,
            None if rgb.disagreement_ms is None else round(rgb.disagreement_ms, 1),
            HAND_LATENCY_MS,
            12.0,
        )
        result[episode.key] = TimeSync(
            episode.key,
            episode.session,
            episode.date,
            episode.episode,
            VERSION,
            "OAK left video PTS as cut (video_time); capture_time keeps every stream's PTS as cut",
            None if source.failed else offsets,
            ["links_failed"] if source.failed else flags,
            rgb.model,
            exo,
            confidence,
            links,
            version=FIT_VERSION,
            hand_latency_ms=HAND_LATENCY_MS,
        )
    return result


def fit_session(episodes: list[FitEpisode]) -> dict[str, TimeSync]:
    """Fit one typed session and return strict, JSON-ready per-episode records."""
    if not episodes or len({(episode.date, episode.session) for episode in episodes}) != 1:
        raise ValueError("fit_session requires one non-empty session")
    pose_guesses: list[float] = []
    for episode in episodes:
        if episode.observation.failed:
            continue
        links = episode.observation.links
        lp, lg, gp = links.lag_left_to_pose, links.lag_left_to_gyro, links.lag_gyro_to_pose
        if lp.lag_s is not None and lp.peak is not None and lp.peak >= 0.5:
            pose_guesses.append(-lp.lag_s * 1000.0)
        if lg.lag_s is not None and gp.lag_s is not None and lg.peak is not None and gp.peak is not None and min(lg.peak, gp.peak) >= 0.5:
            pose_guesses.append(-(lg.lag_s + gp.lag_s) * 1000.0)
    grid = coarse_grid(float(np.median(pose_guesses)) if pose_guesses else None)
    phones = dict.fromkeys(phone for episode in episodes for phone in episode.phones if phone in episode.detections)
    detections: dict[str, dict[str, PhoneDetections]] = {
        phone: {episode.episode: episode.detections[phone] for episode in episodes if phone in episode.phones and phone in episode.detections}
        for phone in phones
    }
    quests = {episode.episode: episode.quest for episode in episodes}
    fits: dict[str, PhoneFit] = {phone: fit_phone(frames, quests, grid) for phone, frames in detections.items()}
    return sync_records(episodes, fits, fill_phone_offsets(episodes, fits))
