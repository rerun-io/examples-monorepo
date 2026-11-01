"""Session exo calibration v3: the hands fit, extended and refined with the wearer's head track.

Every phone that sees the wearer's head gets a pose, including phones that never see the hands. Hands-only poses
seed the face offsets; the joint fit then refines every phone on align's face and hand detections. A phone whose head fit
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
    BODY,
    CLOCK_SUSPECT_GAIN,
    CLOCK_SUSPECT_S,
    DEFAULT_OFFSETS,
    FACE,
    MIN_FACE_PAIRS,
    BodyTrack,
    FaceTrack,
    HeadTrack,
    PhoneReport,
    body_bundle,
    clock_scan,
    face_correspondences,
    face_world,
    find_clock,
    gate_hands,
    joint_bundle,
    residual_summary,
    solve_from_head,
    upright,
    world_up,
)
from selfcap_curate.quest_fit import Correspondences, pairs
from selfcap_curate.records import CatalogHands, SessionEpisode


@dataclass(frozen=True, slots=True)
class HeadCalibration:
    """The v3 session record and its provenance."""

    exo: SessionCalibration
    offsets: Float64[ndarray, "5 3"]
    offsets_source: str
    phones: dict[str, PhoneReport]
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
    """A hands-only pose the head fit could not check (no face, no own hands within the gate): kept on record,
    flagged for the episode so keypoints leaves it out."""
    current = phone.episodes.get(episode) or EpisodeCalibration(phone.n, phone.median_px, False, "flagged")
    return replace(phone, ok=False, episodes={**phone.episodes, episode: replace(current, ok=False, scope="flagged")})


def _views(episode: SessionEpisode) -> tuple[dict[str, FaceTrack], dict[str, Correspondences], dict[str, BodyTrack]]:
    """Each phone's wearer from align's detections: face and body tracks on the aligned clock, and hand pairs.

    Align keeps one person per frame and a few dozen frames per phone, spread over the session; the phones being
    fixed, that is enough to pose them. A phone without a clock has no aligned frames and is left out.
    """
    faces: dict[str, FaceTrack] = {}
    hands: dict[str, Correspondences] = {}
    bodies: dict[str, BodyTrack] = {}
    offsets_ms = episode.sync.offsets_ms or {}
    clocked = {p: d for p, d in episode.detections.items() if offsets_ms.get("exo/" + p) is not None}
    aligned = {p: d.times_s + offsets_ms["exo/" + p] / 1000.0 for p, d in clocked.items()}
    # Align detects the same frame numbers on every phone. The body needs several phones at one instant, so it uses
    # each frame's mean aligned time (the phone clocks differ by tens of milliseconds); face and hands keep their own.
    shared = np.mean(list(aligned.values()), axis=0) if len({len(t) for t in aligned.values()}) == 1 else None
    if shared is None:
        print("HEAD_BODY_SKIP", episode.sync.key, "phones have different detection frames:", {p: len(t) for p, t in aligned.items()}, flush=True)
    for phone, detections in clocked.items():
        faces[phone] = FaceTrack(aligned[phone], detections.xy[:, list(FACE)], detections.scores[:, list(FACE)])
        if shared is not None:
            bodies[phone] = BodyTrack(shared, detections.xy[:, list(BODY)], detections.scores[:, list(BODY)])
        fit = episode.sync.exo_fit.get(phone)
        if fit is not None and fit.tau_phone_to_pose_ms is not None:
            hands[phone] = pairs(detections, episode.quest, fit.tau_phone_to_pose_ms / 1000.0)
    return faces, hands, bodies


BODY_SLACK_PX: float = 2.0
"""A body-refined pose replaces its face-and-hands pose only if neither its face nor its hand median grows by more."""


def _refine_with_body(fitted: dict[str, Camera], faces: dict[str, FaceTrack], gated: dict[str, Correspondences], faces_bad: set[str],
                      shifts: dict[str, float], bodies: dict[str, BodyTrack], track: HeadTrack, offsets: Float64[ndarray, "5 3"],
                      up: Float64[ndarray, "3"]) -> tuple[dict[str, Camera], set[str]]:
    """Pull the accepted poses into agreement on the wearer's body (body_bundle); keep each refined pose that still fits
    its own face and hands within BODY_SLACK_PX and stays upright. Phones on a corrected clock sit out: their detections
    are on the old clock."""
    phones = [p for p in fitted if p not in shifts]
    if len(phones) < 2 or any(p not in bodies for p in phones):
        return fitted, set()
    refined = body_bundle({p: fitted[p] for p in phones}, {p: faces[p] for p in phones if p not in faces_bad},
                          {p: gated[p] for p in phones if p in gated}, {p: bodies[p] for p in phones}, track, offsets)
    kept, moved = dict(fitted), set()
    for phone, camera in refined.items():
        own = [gated[phone]] if phone in gated and len(gated[phone].world) else []
        checks = own if phone in faces_bad else [face_correspondences(faces[phone], track, offsets), *own]
        before = [residual_summary(fitted[phone], data)[1] for data in checks]
        after = [residual_summary(camera, data)[1] for data in checks]
        if upright(camera, up) and all(a is not None and b is not None and a <= b + BODY_SLACK_PX for a, b in zip(after, before, strict=True)):
            kept[phone] = camera
            moved.add(phone)
    return kept, moved


def _shifted(face: FaceTrack, shift: float) -> FaceTrack:
    return face if shift == 0.0 else FaceTrack(face.times_s + shift, face.xy, face.scores)


def calibrate_with_head(exo: SessionCalibration, episode: SessionEpisode, track: HeadTrack,
                        focals: dict[str, float], clock_ms: dict[str, float], provisional: frozenset[str] = frozenset()) -> HeadCalibration:
    """Extend and refine one episode's hands-only calibration with the head track (single-episode sessions).

    The observations are align's detections (``episode.detections``) on ``episode.sync``'s clocks; ``clock_ms`` holds
    each phone's offset as used there, and ``provisional`` phones had no clock and carry a guessed one. Their clock is
    found from the face track alone; every other phone's clock is checked the same way, and a decisive shift is
    corrected. Corrected phones lose their hand pairs (paired on the old clock).
    """
    hand_cameras = {p: c.camera for p, c in exo.phones.items() if c.ok and c.camera is not None and p not in provisional}
    default_focal = float(np.median(list(focals.values()))) if focals else 900.0
    observed_faces, observed_hands, bodies = _views(episode)
    offsets = DEFAULT_OFFSETS.copy()
    source = "template (d31 label-refined cameras), fixed"
    # A hands-only pose with too few face pairs to check is solved from the head like an unposed phone.
    blind = {p for p in hand_cameras if p not in observed_faces or len(face_correspondences(observed_faces[p], track, offsets).world) < MIN_FACE_PAIRS}
    shifts: dict[str, float] = {}
    for phone in sorted(provisional & set(observed_faces)):
        found = find_clock(observed_faces[phone], track, offsets, focals.get(phone, default_focal))
        if found is not None:
            shifts[phone] = found
    cameras: dict[str, Camera] = {p: c for p, c in hand_cameras.items() if p not in blind}
    for phone in observed_faces:
        if phone in cameras or (phone in provisional and phone not in shifts):
            continue
        camera = solve_from_head(face_correspondences(_shifted(observed_faces[phone], shifts.get(phone, 0.0)), track, offsets), focals.get(phone, default_focal))
        if camera is not None:
            cameras[phone] = camera
    if not cameras:
        return HeadCalibration(exo, offsets, source, {}, {})

    def fit(cameras: dict[str, Camera]) -> tuple[dict[str, Camera], dict[str, FaceTrack], dict[str, Correspondences]]:
        views = {p: _shifted(v, shifts.get(p, 0.0)) for p, v in observed_faces.items()}
        hand_pairs = {p: h for p, h in observed_hands.items() if p not in shifts}
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
    # pose from its own hands instead: with the offsets fixed, each phone's fit is independent.
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
    # Face and hands pose each phone on its own, well near the head and hands; the body the phones share pulls them into
    # agreement far below (legs, elbows), within what their own face and hands allow.
    fitted, body_moved = _refine_with_body(fitted, faces, gated, faces_bad, shifts, bodies, track, offsets, up)
    # A clock correction stands only if the corrected phone fits its face and the session's other phones fit theirs:
    # otherwise the scan measured a template that does not hold here. Phones with no clock at all keep theirs.
    others = [v for p, v in face_px.items() if p not in shifts and v is not None]
    template_holds = not others or float(np.median(others)) <= 20.0
    corrections = {p: round(clock_ms[p] + 1000.0 * shift, 1) for p, shift in shifts.items()
                   if p in clock_ms and (p in provisional or (template_holds and face_px.get(p) is not None and face_px[p] <= 15.0))}
    for phone in sorted(set(episode.detections) | set(exo.phones)):
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
        method += " + body" if phone in body_moved else ""
        reports[phone] = PhoneReport(method, face_n, face_med, hand_n, hand_px, shift, before_px, best_shift, gain, corrections.get(phone))
    return HeadCalibration(SessionCalibration(exo.session, phones, exo.body_median_px), offsets, source, reports, corrections)


def face_points(track: HeadTrack, offsets: Float64[ndarray, "5 3"], times: Float64[ndarray, "n"]) -> Float64[ndarray, "n 5 3"]:
    """World face points, for display and checks."""
    return face_world(track, offsets, times)
