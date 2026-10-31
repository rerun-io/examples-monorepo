"""Session exo calibration v3: the hands fit, extended and refined with the wearer's head track.

Every phone that sees the wearer's head gets a pose, including phones that never see the hands. Hands-only poses
seed the face offsets; the joint fit then refines every phone on dense face and hand pairs. A phone whose head fit
does not reach ACCEPT_FACE_PX keeps its hands-only pose (or stays unposed).
"""

from dataclasses import dataclass, field, replace

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from selfcap_curate.exo_calibration import Camera, EpisodeCalibration, PhoneCalibration, SessionCalibration, solve_camera
from selfcap_curate.head_anchor import (
    ACCEPT_FACE_PX,
    CLOCK_SUSPECT_GAIN,
    CLOCK_SUSPECT_S,
    DEFAULT_OFFSETS,
    FACE,
    MIN_FACE_PAIRS,
    FaceTrack,
    HeadTrack,
    PhoneReport,
    chosen,
    clock_scan,
    estimate_offsets,
    face_correspondences,
    face_world,
    find_clock,
    gate_hands,
    joint_bundle,
    residual_summary,
    select_person,
    solve_from_head,
    upright,
    world_up,
)
from selfcap_curate.keypoints import Keypoints2D
from selfcap_curate.quest_fit import Correspondences, pairs
from selfcap_curate.records import CatalogHands, PhoneDetections, SessionEpisode


@dataclass(frozen=True, slots=True)
class HeadCalibration:
    """The v3 session record and its provenance."""

    exo: SessionCalibration
    offsets: Float64[ndarray, "5 3"]
    offsets_source: str
    phones: dict[str, PhoneReport]
    measured_offsets: Float64[ndarray, "5 3"] | None = None
    """Offsets triangulated from the hands-only phones, before the joint fit."""
    clock_corrections_ms: dict[str, float] = field(default_factory=dict)
    """Phone offsets (ms) the head track finds where the phone's clock was missing or decisively wrong."""


RUB_TO_RDF: Float64[ndarray, "3 3"] = np.diag([1.0, -1.0, -1.0])
"""Turns an orientation stored with x right, y up, z back (OpenXR camera axes) into RDF."""


def head_track(source: CatalogHands, quest_pose_ms: float) -> HeadTrack:
    """The Quest head pose (left-eye origin, head-to-world xyzw) moved onto the aligned clock, in RDF axes.

    Some captures (2025-12-11 and 12-12 in the hard set) store the eye orientation with y up and z back. A worn headset
    never spends a session upside down, so a camera "down" axis that points up the Y-up world on median marks that
    convention, and the track is turned back to RDF.
    """
    valid = np.isfinite(source.head_positions).all(1) & (np.linalg.norm(source.head_quaternions, axis=1) > 0.5)
    times = source.head_times_s[valid] + quest_pose_ms / 1000.0
    order = np.flatnonzero(np.r_[True, np.diff(times) > 0.0])
    rotations = Rotation.from_quat(source.head_quaternions[valid][order]).as_matrix()
    if np.median(rotations[:, 1, 1]) > 0.0:
        return HeadTrack(times[order], rotations @ RUB_TO_RDF, source.head_positions[valid][order], convention="RUB")
    return HeadTrack(times[order], rotations, source.head_positions[valid][order])


def unverified(phone: PhoneCalibration, episode: str) -> PhoneCalibration:
    """A hands-only pose the head fit could not check (no face, no dense hands within the gate): kept on record,
    flagged for the episode so keypoints leaves it out."""
    current = phone.episodes.get(episode) or EpisodeCalibration(phone.n, phone.median_px, False, "flagged")
    return replace(phone, ok=False, episodes={**phone.episodes, episode: replace(current, ok=False, scope="flagged")})


def _views(dense: Keypoints2D, track: HeadTrack, cameras: dict[str, Camera], offsets: Float64[ndarray, "5 3"] | None,
           episode: SessionEpisode) -> tuple[dict[str, FaceTrack], dict[str, Correspondences]]:
    """Each phone's wearer: face track and dense hand pairs; posed phones pick the person at the projected head."""
    faces: dict[str, FaceTrack] = {}
    hands: dict[str, Correspondences] = {}
    offsets_ms = episode.sync.offsets_ms or {}
    centre = np.nanmean(offsets, axis=0) if offsets is not None and np.isfinite(offsets).any() else np.zeros(3)
    rotations, positions, valid = track.at(dense.times)
    head_point = np.einsum("nij,j->ni", rotations, centre) + positions
    head_point[~valid] = np.nan
    for phone, view in dense.views.items():
        expected = None
        if phone in cameras:
            projected = cameras[phone].project(np.nan_to_num(head_point))
            expected = np.where((projected[:, 2:3] > 0.0) & valid[:, None], projected[:, :2], np.nan)
        choice = select_person(view.boxes, view.box_scores, view.xy, view.scores, expected)
        xy = chosen(view.xy, choice, np.nan)
        scores = chosen(view.scores, choice, 0.0)
        faces[phone] = FaceTrack(dense.times, xy[:, list(FACE)], scores[:, list(FACE)])
        fit = episode.sync.exo_fit.get(phone)
        offset = offsets_ms.get("exo/" + phone)
        if fit is not None and fit.tau_phone_to_pose_ms is not None and offset is not None:
            capture = dense.times - offset / 1000.0
            hands[phone] = pairs(PhoneDetections(capture, xy, scores), episode.quest, fit.tau_phone_to_pose_ms / 1000.0)
    return faces, hands


def _shifted(face: FaceTrack, shift: float) -> FaceTrack:
    return face if shift == 0.0 else FaceTrack(face.times_s + shift, face.xy, face.scores)


def calibrate_with_head(exo: SessionCalibration, episode: SessionEpisode, dense: Keypoints2D, track: HeadTrack,
                        focals: dict[str, float], clock_ms: dict[str, float], provisional: frozenset[str] = frozenset()) -> HeadCalibration:
    """Extend and refine one episode's hands-only calibration with the head track (single-episode sessions).

    ``clock_ms`` holds each phone's offset as used for ``dense``; ``provisional`` phones had no clock and were
    detected on a guessed one. Their clock is found from the face track alone; every other phone's clock is checked
    the same way, and a decisive shift is corrected. Corrected phones lose their hand pairs (paired on the old clock).
    """
    hand_cameras = {p: c.camera for p, c in exo.phones.items() if c.ok and c.camera is not None and p not in provisional}
    default_focal = float(np.median(list(focals.values()))) if focals else 900.0
    faces, hands = _views(dense, track, hand_cameras, None, episode)
    measured = estimate_offsets(hand_cameras, faces, track)  # diagnostic only: hands-only poses are biased at the head
    offsets = DEFAULT_OFFSETS.copy()
    source = "template (d31 label-refined cameras), fixed"
    # A hands-only pose that projects the head where no face is found picked nobody: pick the wearer without it and
    # solve that phone from the head like an unposed one.
    blind = {p for p in hand_cameras if len(face_correspondences(faces[p], track, offsets).world) < MIN_FACE_PAIRS}
    if blind:
        unguided, _ = _views(dense, track, {}, None, episode)
        faces.update({p: unguided[p] for p in blind})
    shifts: dict[str, float] = {}
    for phone in sorted(provisional & set(dense.views)):
        found = find_clock(faces[phone], track, offsets, focals.get(phone, default_focal))
        if found is not None:
            shifts[phone] = found
    cameras: dict[str, Camera] = {p: c for p, c in hand_cameras.items() if p not in blind}
    for phone in dense.views:
        if phone in cameras or (phone in provisional and phone not in shifts):
            continue
        camera = solve_from_head(face_correspondences(_shifted(faces[phone], shifts.get(phone, 0.0)), track, offsets), focals.get(phone, default_focal))
        if camera is not None:
            cameras[phone] = camera
    if not cameras:
        return HeadCalibration(exo, offsets, source, {}, measured, {})

    def fit(cameras: dict[str, Camera]) -> tuple[dict[str, Camera], dict[str, FaceTrack], dict[str, Correspondences]]:
        views, hand_pairs = _views(dense, track, cameras, offsets, episode)
        views = {p: _shifted(v, shifts.get(p, 0.0)) for p, v in views.items()}
        hand_pairs = {p: h for p, h in hand_pairs.items() if p not in shifts}
        gated = {p: gate_hands(cameras[p], hand_pairs[p]) for p in cameras if p in hand_pairs}
        fitted, _ = joint_bundle(cameras, {p: views[p] for p in cameras}, gated, track, offsets, fit_offsets=False)
        gated = {p: gate_hands(fitted[p], hand_pairs[p]) for p in fitted if p in hand_pairs}
        fitted, _ = joint_bundle(fitted, {p: views[p] for p in fitted}, gated, track, offsets, fit_offsets=False)
        return fitted, views, gated

    fitted, faces, gated = fit(cameras)
    faces_bad: set[str] = set()
    scans: dict[str, tuple[float | None, float | None]] = {}
    suspect = False
    for phone, camera in fitted.items():
        if phone in shifts:
            continue
        best, gain = clock_scan(faces[phone], track, offsets, camera)
        scans[phone] = (best, gain)
        if best is not None and gain is not None and abs(best) >= CLOCK_SUSPECT_S and gain >= CLOCK_SUSPECT_GAIN:
            fine, _ = clock_scan(faces[phone], track, offsets, camera, np.arange(best - 0.3, best + 0.3001, 0.04))
            shifts[phone] = best if fine is None else fine
            moved = face_correspondences(_shifted(faces[phone], shifts[phone]), track, offsets)
            cameras[phone] = solve_from_head(moved, float(camera.parameters[6]), seed=camera) or camera
            suspect = True
    if suspect:
        fitted, faces, gated = fit({p: cameras.get(p, c) if p in shifts else c for p, c in fitted.items()})
    phones: dict[str, PhoneCalibration] = dict(exo.phones)
    reports: dict[str, PhoneReport] = {}
    face_px = {p: residual_summary(c, face_correspondences(faces[p], track, offsets))[1] for p, c in fitted.items()}
    # A phone whose face does not fit the template (back views, or a head stream that disagrees with the hands) keeps a
    # pose from its own dense hands instead: with the offsets fixed, each phone's fit is independent.
    up = world_up(track)
    for phone, camera in list(fitted.items()):
        if face_px[phone] is not None and face_px[phone] <= ACCEPT_FACE_PX and upright(camera, up):
            continue
        own = gated.get(phone)
        if own is not None and len(own.world) >= 60 and phone not in shifts:
            refit = solve_camera(own, float(camera.parameters[6]), seed=camera)
            if refit is not None and upright(refit, up):
                fitted[phone] = refit
                faces_bad.add(phone)
                continue
        del fitted[phone]
    # A clock correction stands only if the corrected phone fits its face and the session's other phones fit theirs:
    # otherwise the scan measured a template that does not hold here. Phones with no clock at all keep theirs.
    others = [v for p, v in face_px.items() if p not in shifts and v is not None]
    template_holds = not others or float(np.median(others)) <= 20.0
    corrections = {p: round(clock_ms[p] + 1000.0 * shift, 1) for p, shift in shifts.items()
                   if p in clock_ms and (p in provisional or (template_holds and face_px.get(p) is not None and face_px[p] <= 15.0))}
    for phone in sorted(set(dense.views) | set(exo.phones)):
        before = hand_cameras.get(phone)
        camera = fitted.get(phone)
        best_shift, gain = scans.get(phone, (shifts.get(phone), None))
        if camera is None:
            if phone in phones and phones[phone].camera is not None:
                phones[phone] = unverified(phones[phone], episode.sync.episode)
            reports[phone] = PhoneReport("hands (unverified, flagged)" if before is not None else "unposed", 0, None, 0, None, None, None,
                                         best_shift, gain, corrections.get(phone))
            continue
        face = face_correspondences(faces[phone], track, offsets)
        face_n, face_med = residual_summary(camera, face)
        own_hands = gated.get(phone, Correspondences(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0)))
        hand_n, hand_px = residual_summary(camera, own_hands)
        before_px = None if before is None else residual_summary(before, face)[1]
        accepted = phone in faces_bad or (face_med is not None and face_n >= 30 and face_med <= ACCEPT_FACE_PX)
        shift = None if before is None else float(np.linalg.norm(camera.world_T_cam[:3, 3] - before.world_T_cam[:3, 3]))
        if not accepted:
            if phone in phones and phones[phone].camera is not None:
                phones[phone] = unverified(phones[phone], episode.sync.episode)
            reports[phone] = PhoneReport("hands (unverified, flagged)" if before is not None else "unposed", face_n, face_med, hand_n, hand_px,
                                         shift, before_px, best_shift, gain, corrections.get(phone))
            continue
        residual = (camera.residuals(own_hands) if phone in faces_bad else
                    np.r_[camera.residuals(face), camera.residuals(own_hands) if len(own_hands.world) else np.zeros(0)])
        finite = residual[np.isfinite(residual)]
        phones[phone] = PhoneCalibration(True, camera, float(np.median(finite)), float(np.percentile(finite, 90)), len(residual))
        method = "dense hands (face rejected)" if phone in faces_bad else ("hands+head" if before is not None else "head")
        reports[phone] = PhoneReport(method, face_n, face_med, hand_n, hand_px, shift, before_px, best_shift, gain, corrections.get(phone))
    return HeadCalibration(SessionCalibration(exo.session, phones, exo.body_median_px), offsets, source, reports, measured, corrections)


def face_points(track: HeadTrack, offsets: Float64[ndarray, "5 3"], times: Float64[ndarray, "n"]) -> Float64[ndarray, "n 5 3"]:
    """World face points, for display and checks."""
    return face_world(track, offsets, times)
