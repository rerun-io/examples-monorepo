"""Per-session CPU camera/time fit, ported from frozen fit_session.py."""

import math
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from selfcap_curate.quest_fit import Correspondences, FitMetrics, cost, curve_metrics, pairs, robust_fit
from selfcap_curate.records import HAND_LATENCY_MS, EpisodeKey, HandTrack, Links, PhoneDetections, Side


@dataclass(frozen=True, slots=True)
class EpisodeLinks:
    """Rotation links plus the clock/window metadata the session model needs."""

    key: str
    """Episode key."""
    links: Links
    """Coarse and optional fine gyro-referenced links."""
    window_s: tuple[float, float]
    """Motion measurement interval on the episode clock."""
    t_start_session_s: float | None
    """Episode start in session seconds (native OAK frame index / 24)."""
    failed: bool = False
    """Link measurement failed for this episode."""


@dataclass(frozen=True, slots=True)
class FitEpisode:
    """One episode's typed CPU-fit inputs."""

    session: str
    """Full session UUID."""
    observation: EpisodeLinks
    """Rotation measurements and original measurement window."""
    phones: tuple[str, ...]
    """Full phone names, including devices with no detections."""
    detections: dict[str, PhoneDetections]
    """Presented-PTS detections from catalog inference or a stored stream."""
    quest: dict[Side, HandTrack]
    """Raw-clock Quest hands, read from the catalog and filtered once."""

    @property
    def key(self) -> str:
        """Full episode key."""
        return self.observation.key

    @property
    def episode(self) -> str:
        """Episode label within the session."""
        return EpisodeKey.parse(self.key).episode

    @property
    def date(self) -> str:
        """Session date."""
        return EpisodeKey.parse(self.key).date


PhoneFit: TypeAlias = dict[str, FitMetrics]
NeverFitChoice: TypeAlias = Literal["same_clock", "own_weak_fit"]


@dataclass(frozen=True, slots=True)
class PhoneFill:
    """Robust additive clock model tau_phone,episode = T_episode + c_phone."""

    episode_time_ms: dict[str, float | None]
    """Common episode clocks; null where no trusted phone constrains the clock."""
    phone_constants_ms: dict[str, float]
    """Measured phone constants, including selected never-fit hypotheses."""
    never_fit: dict[str, NeverFitChoice]
    """Explicit unverified/weak clock choices."""


def coarse_grid(quest_pose_ms: float | None) -> Float64[ndarray, "n"]:
    """Search ±2.5 s about both Quest-clock and OAK-clock phone hypotheses.

    The dual centre guards sessions cut with Quest 1–7 s away from OAK; a
    single zero-centred grid could otherwise select an apparently strong edge.
    """
    centre: float = (HAND_LATENCY_MS - quest_pose_ms) / 1000.0 if quest_pose_ms is not None and math.isfinite(quest_pose_ms) else 0.0
    return np.arange(min(-2.5, centre - 2.5), max(2.5, centre + 2.5) + 1e-9, 0.020)


def fit_phone(detections: dict[str, PhoneDetections], quests: dict[str, dict[Side, HandTrack]], grid_coarse: Float64[ndarray, "n"]) -> PhoneFit:
    """Fit one camera per session and alternate three per-episode lag refinements.

    A shared camera guards against explaining a time error with a different
    camera in every episode. Fine searches are ±60 ms at 2 ms spacing with
    parabolic refinement. Coarse/fine edge minima remain untrusted.
    """
    episodes: list[str] = [episode for episode in detections if len(detections[episode].times_s)]
    if not episodes:
        return {}

    def stacked(taus: dict[str, float]) -> Correspondences:
        return Correspondences.concat([pairs(detections[episode], quests[episode], taus[episode]) for episode in episodes])

    coarse_values: list[float] = []
    for tau in grid_coarse:
        data: Correspondences = stacked(dict.fromkeys(episodes, float(tau)))
        camera = robust_fit(data)
        coarse_values.append(cost(camera, data))
    coarse = np.asarray(coarse_values, dtype=np.float64)
    if not np.isfinite(coarse).any():
        return {}
    tau0: float = float(grid_coarse[int(np.argmin(coarse))])
    taus: dict[str, float] = dict.fromkeys(episodes, tau0)
    result: dict[str, FitMetrics] = {}
    for _ in range(3):
        camera = robust_fit(stacked(taus))
        if camera is None:
            return {}
        for episode in episodes:
            grid = np.arange(taus[episode] - 0.060, taus[episode] + 0.060 + 1e-9, 0.002)
            values = np.array([cost(camera, pairs(detections[episode], quests[episode], float(tau))) for tau in grid], dtype=np.float64)
            metrics: FitMetrics = curve_metrics(grid, values)
            if metrics.tau_ms is not None:
                taus[episode] = metrics.tau_ms / 1000.0
            points: int = len(pairs(detections[episode], quests[episode], taus[episode]).world)
            edge: bool = bool(
                metrics.tau_ms is None or abs(metrics.tau_ms / 1000.0 - grid[0]) < 0.004 or abs(metrics.tau_ms / 1000.0 - grid[-1]) < 0.004
            )
            result[episode] = replace(metrics, points=points, edge=edge)
    if int(np.nanargmin(coarse)) in (0, len(coarse) - 1):
        result = {episode: replace(metrics, edge=True) for episode, metrics in result.items()}
    return result


def good_fit(metrics: FitMetrics) -> bool:
    """Trust ≥60 pairs, ≤14 px cost, ≤40 ms basin, and no edge hit.

    These guards reject weak detections, broad temporal ambiguity, and minima
    that lie beyond the searched range before they can anchor other phones.
    """
    return (
        metrics.tau_ms is not None
        and math.isfinite(metrics.tau_ms)
        and metrics.points >= 60
        and metrics.cmin_px is not None
        and metrics.cmin_px <= 14.0
        and metrics.basin_halfwidth_ms is not None
        and metrics.basin_halfwidth_ms <= 40.0
        and not metrics.edge
    )


def fill_phone_offsets(episodes: list[FitEpisode], fits: dict[str, PhoneFit]) -> PhoneFill:
    """Five median alternations fit tau = T_episode + c_phone using trusted fits.

    This guards missing and outlying episodes without inventing a measurement.
    A never-good phone has no measured constant: compare same-clock versus its
    weak clock with one session camera per hypothesis. Require ≥3 px cost gain
    for the weak clock, guarding ambiguous improvements (29-phone pixel check).
    """
    measurements: dict[str, dict[str, float]] = {
        phone: {ep: metrics.tau_ms for ep, metrics in fit.items() if good_fit(metrics) and metrics.tau_ms is not None} for phone, fit in fits.items()
    }
    constants: dict[str, float] = dict.fromkeys(measurements, 0.0)
    common: dict[str, float | None] = {}
    for _ in range(5):
        for episode in episodes:
            values = [measurements[phone][episode.episode] - constants[phone] for phone in measurements if episode.episode in measurements[phone]]
            common[episode.episode] = float(np.median(values)) if values else None
        for phone in measurements:
            values = [value - centre for episode, value in measurements[phone].items() if (centre := common.get(episode)) is not None]
            constants[phone] = float(np.median(values)) if values else 0.0
    never: dict[str, NeverFitChoice] = {}
    by_id: dict[str, FitEpisode] = {episode.episode: episode for episode in episodes}
    anchors: dict[str, float] = {ep: value for ep, value in common.items() if value is not None}
    for phone, measured in measurements.items():
        if measured:
            continue
        own: dict[str, float] = {
            ep: metrics.tau_ms
            for ep, metrics in fits[phone].items()
            if metrics.tau_ms is not None and ep in anchors and phone in by_id[ep].detections
        }
        if not own:
            never[phone] = "same_clock"
            continue
        own_constant: float = float(np.median([tau - anchors[ep] for ep, tau in own.items()]))
        costs: dict[str, float] = {}
        for name, constant in (("same_clock", 0.0), ("own_weak_fit", own_constant)):
            data = Correspondences.concat([pairs(by_id[ep].detections[phone], by_id[ep].quest, (anchors[ep] + constant) / 1000.0) for ep in own])
            camera = robust_fit(data)
            costs[name] = cost(camera, data)
        choose_own: bool = costs["same_clock"] - costs["own_weak_fit"] >= 3.0
        constants[phone] = own_constant if choose_own else 0.0
        never[phone] = "own_weak_fit" if choose_own else "same_clock"
    return PhoneFill(common, constants, never)
