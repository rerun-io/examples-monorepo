"""Exo cameras anchored on the wearer's head track: dense face keypoints against the ego device's SLAM head pose.

A device worn on the head with its own SLAM (Quest, Aria) is a moving, known 3D target in every exo view. The face
keypoints a 2D detector reports (nose, eyes, ears) sit at fixed offsets in that device's frame, so each exo frame that
sees the head gives five 2D–3D pairs. Over an episode the head wanders and turns, which conditions a static camera's
pose far better than the hands a box can hide for the whole take. The offsets are measured from phones that are
already calibrated, or fitted jointly with the cameras; the head track itself is fixed truth, so there is no gauge.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from selfcap_curate.exo_calibration import Camera, solve_camera
from selfcap_curate.quest_fit import Correspondences

FACE: tuple[int, ...] = (0, 1, 2, 3, 4)
"""COCO nose, left eye, right eye, left ear, right ear."""
MIN_FACE_SCORE: float = 0.3
"""Detector score gate, as the hand pairs and triangulation use."""
MIN_FACE_PAIRS: int = 30
"""Fewer pairs than this cannot seed a pose."""
TRIANGULATION_GATE_PX: float = 15.0
"""A triangulated face point is kept for the offsets only if every supporting view agrees to this many pixels."""
MIN_OFFSET_FRAMES: int = 15
"""Frames a face joint needs before its offset is measured rather than borrowed."""
HEAD_GAP_S: float = 0.1
"""Largest head-pose gap that interpolation bridges, as read_anchors uses."""


@dataclass(frozen=True, slots=True)
class HeadTrack:
    """The ego device's head pose on the aligned clock."""

    times_s: Float64[ndarray, "n"]
    """Aligned (video_time) seconds, increasing."""
    rotations: Float64[ndarray, "n 3 3"]
    """Head-to-world rotations."""
    positions: Float64[ndarray, "n 3"]
    """Head origin in world metres."""
    convention: str = "RDF"
    """Axes the source stored the orientation in; the rotations here are always RDF (x right, y down, z forward)."""

    def at(self, times: Float64[ndarray, "m"]) -> tuple[Float64[ndarray, "m 3 3"], Float64[ndarray, "m 3"], Bool[ndarray, "m"]]:
        """Slerp/lerp between neighbours no more than HEAD_GAP_S apart; invalid rows are identity at the origin."""
        stamps = self.times_s
        right = np.clip(np.searchsorted(stamps, times), 1, len(stamps) - 1)
        left = right - 1
        gap = stamps[right] - stamps[left]
        valid = (times >= stamps[0]) & (times <= stamps[-1]) & (gap > 0.0) & (gap <= HEAD_GAP_S)
        clipped = np.clip(times, stamps[0], stamps[-1])
        rotations = Slerp(stamps, Rotation.from_matrix(self.rotations))(clipped).as_matrix()
        alpha = ((clipped - stamps[left]) / np.maximum(gap, 1e-12))[:, None]
        positions = (1.0 - alpha) * self.positions[left] + alpha * self.positions[right]
        rotations[~valid], positions[~valid] = np.eye(3), 0.0
        return rotations, positions, valid


@dataclass(frozen=True, slots=True)
class FaceTrack:
    """One exo view's selected wearer: face keypoints on the dense aligned grid."""

    times_s: Float64[ndarray, "n"]
    """Aligned seconds of each detection frame."""
    xy: Float64[ndarray, "n 5 2"]
    """FACE keypoints in pixels; NaN when absent."""
    scores: Float64[ndarray, "n 5"]
    """Detector scores; zero when absent."""


def face_world(track: HeadTrack, offsets: Float64[ndarray, "5 3"], times: Float64[ndarray, "n"]) -> Float64[ndarray, "n 5 3"]:
    """Face points in world metres at ``times``; NaN outside the head track or for unknown offsets."""
    rotations, positions, valid = track.at(times)
    world = np.einsum("nij,kj->nki", rotations, offsets) + positions[:, None]
    world[~valid] = np.nan
    return world


def face_correspondences(face: FaceTrack, track: HeadTrack, offsets: Float64[ndarray, "5 3"]) -> Correspondences:
    """Scored face detections paired with their world points; score is the pair weight."""
    world = face_world(track, offsets, face.times_s)
    keep = (face.scores >= MIN_FACE_SCORE) & np.isfinite(face.xy).all(-1) & np.isfinite(world).all(-1)
    return Correspondences(world[keep], face.xy[keep], face.scores[keep])


def solve_from_head(data: Correspondences, focal: float, *, seed: Camera | None = None) -> Camera | None:
    """EPnP RANSAC seed then robust refinement at a fixed focal; None without enough pairs."""
    if len(data.world) < MIN_FACE_PAIRS:
        return None
    return solve_camera(data, focal, seed=seed)


def _projection(camera: Camera) -> Float64[ndarray, "3 4"]:
    p = camera.parameters
    return np.array([[p[6], 0.0, p[7]], [0.0, p[6], p[8]], [0.0, 0.0, 1.0]]) @ np.c_[Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6]]


def estimate_offsets(cameras: Mapping[str, Camera], faces: Mapping[str, FaceTrack], track: HeadTrack) -> Float64[ndarray, "5 3"] | None:
    """Triangulate each face joint from >=2 calibrated views, move it into the head frame and take the median.

    Views must share one time grid. A joint seen in fewer than MIN_OFFSET_FRAMES agreeing frames stays NaN; None when
    no joint is measured.
    """
    phones = [p for p in cameras if p in faces]
    if len(phones) < 2:
        return None
    times = faces[phones[0]].times_s
    if any(not np.array_equal(faces[p].times_s, times) for p in phones):
        raise ValueError("face tracks must share one time grid")
    rotations, positions, valid = track.at(times)
    matrices = {p: _projection(cameras[p]) for p in phones}
    samples: list[list[Float64[ndarray, "3"]]] = [[] for _ in FACE]
    for frame in np.flatnonzero(valid):
        for joint in range(len(FACE)):
            views = [p for p in phones if faces[p].scores[frame, joint] >= MIN_FACE_SCORE and np.isfinite(faces[p].xy[frame, joint]).all()]
            if len(views) < 2:
                continue
            rows = []
            for p in views:
                u, v = faces[p].xy[frame, joint]
                rows.extend([u * matrices[p][2] - matrices[p][0], v * matrices[p][2] - matrices[p][1]])
            homogeneous = np.linalg.svd(np.asarray(rows))[2][-1]
            if abs(homogeneous[3]) < 1e-12:
                continue
            point = homogeneous[:3] / homogeneous[3]
            errors = [np.linalg.norm(cameras[p].project(point[None])[0, :2] - faces[p].xy[frame, joint]) for p in views]
            depths = [cameras[p].project(point[None])[0, 2] for p in views]
            if max(errors) <= TRIANGULATION_GATE_PX and min(depths) > 0.0:
                samples[joint].append(rotations[frame].T @ (point - positions[frame]))
    offsets = np.full((len(FACE), 3), np.nan)
    for joint, points in enumerate(samples):
        if len(points) >= MIN_OFFSET_FRAMES:
            offsets[joint] = np.median(np.asarray(points), axis=0)
    return None if np.isnan(offsets).all() else offsets


def joint_bundle(
    cameras: Mapping[str, Camera],
    faces: Mapping[str, FaceTrack],
    hands: Mapping[str, Correspondences],
    track: HeadTrack,
    offsets: Float64[ndarray, "5 3"],
    *,
    max_nfev: int = 100,
    fit_offsets: bool = True,
) -> tuple[dict[str, Camera], Float64[ndarray, "5 3"]]:
    """Fit every camera pose and the face offsets to face and hand pairs; intrinsics and the head track stay fixed.

    Cauchy loss at 6 px, as the hand calibration uses. Offsets with no finite start are left out of the face terms.
    """
    phones = sorted(cameras)
    known = np.isfinite(offsets).all(1)
    free = known if fit_offsets else np.zeros(len(known), dtype=bool)
    head = {p: track.at(faces[p].times_s) if p in faces else None for p in phones}

    def unpack(values: Float64[ndarray, "k"]) -> tuple[dict[str, Camera], Float64[ndarray, "5 3"]]:
        poses = values[: 6 * len(phones)].reshape(-1, 6)
        fitted = {p: Camera(np.r_[poses[i], cameras[p].parameters[6:]]) for i, p in enumerate(phones)}
        rows = offsets.copy()
        rows[free] = values[6 * len(phones) :].reshape(-1, 3)
        return fitted, rows

    def residual(values: Float64[ndarray, "k"]) -> Float64[ndarray, "m"]:
        fitted, rows = unpack(values)
        parts = []
        for p in phones:
            camera = fitted[p]
            if p in faces:
                rotations, positions, valid = head[p]
                world = np.einsum("nij,kj->nki", rotations, rows) + positions[:, None]
                face = faces[p]
                keep = valid[:, None] & known[None] & (face.scores >= MIN_FACE_SCORE) & np.isfinite(face.xy).all(-1)
                uv = camera.project(world[keep])
                r = (uv[:, :2] - face.xy[keep]) * np.sqrt(face.scores[keep])[:, None]
                r[uv[:, 2] <= 0.0] = 1e3
                parts.append(r.ravel())
            if p in hands and len(hands[p].world):
                uv = camera.project(hands[p].world)
                r = (uv[:, :2] - hands[p].image) * np.sqrt(hands[p].weights)[:, None]
                r[uv[:, 2] <= 0.0] = 1e3
                parts.append(r.ravel())
        return np.concatenate(parts) if parts else np.zeros(0)

    start = np.r_[np.concatenate([cameras[p].parameters[:6] for p in phones]), offsets[free].ravel()]
    result = least_squares(residual, start, loss="cauchy", f_scale=6.0, x_scale="jac", max_nfev=max_nfev, method="trf")
    return unpack(result.x)


BODY: tuple[int, ...] = tuple(range(5, 17))
"""COCO shoulders, elbows, wrists, hips, knees and ankles: the joints that tie the phones to each other."""
MIN_BODY_SCORE: float = 0.5
"""Body detections below this score do not join a body point."""
BODY_GATE_PX: float = 25.0
"""A view joins a body point only within this many pixels: wider than the 8 px keypoints gate, because the poses it
corrects disagree by 9-14 px on joints far from the head (7ffcbc7e legs, 4b1b6cdc elbow)."""
MIN_BODY_RAY_DEG: float = 5.0
"""Body points whose agreeing rays all meet at a shallower angle carry no depth and are left out, as in keypoints."""
MAX_BODY_PAIRS: int = 1200
"""Body pairs per phone and round, evenly spaced: about as many as the gated hand pairs, so neither dominates."""
BODY_ROUNDS: int = 3
"""Triangulate-then-refit rounds."""


@dataclass(frozen=True, slots=True)
class BodyTrack:
    """One exo view's selected wearer: BODY keypoints on the dense aligned grid shared by every view."""

    times_s: Float64[ndarray, "n"]
    """Aligned seconds of each detection frame."""
    xy: Float64[ndarray, "n j 2"]
    """BODY keypoints in pixels; NaN when absent."""
    scores: Float64[ndarray, "n j"]
    """Detector scores; zero when absent."""


def triangulate_bodies(cameras: Mapping[str, Camera], bodies: Mapping[str, BodyTrack], gate_px: float = BODY_GATE_PX) -> Float64[ndarray, "n j 3"]:
    """Each frame's body joints from the views that see them: DLT, drop views beyond ``gate_px``, refit.

    NaN where fewer than two views agree, or where the agreeing rays meet at less than MIN_BODY_RAY_DEG.
    """
    phones = [p for p in cameras if p in bodies]
    if not phones:
        return np.zeros((0, len(BODY), 3))
    times = bodies[phones[0]].times_s
    if any(not np.array_equal(bodies[p].times_s, times) for p in phones):
        raise ValueError("body tracks must share one time grid")
    xy = np.stack([bodies[p].xy for p in phones])  # v n j 2
    use = np.stack([bodies[p].scores >= MIN_BODY_SCORE for p in phones]) & np.isfinite(xy).all(-1)
    xy = np.nan_to_num(xy)
    matrices = np.stack([_projection(cameras[p]) for p in phones])  # v 3 4
    rows = np.stack([xy[..., :1] * matrices[:, None, None, 2] - matrices[:, None, None, 0],
                     xy[..., 1:] * matrices[:, None, None, 2] - matrices[:, None, None, 1]], axis=-2)  # v n j 2 4
    centres = np.stack([cameras[p].world_T_cam[:3, 3] for p in phones])

    def solve(mask: Bool[ndarray, "v n j"]) -> Float64[ndarray, "n j 3"]:
        weighted = rows * mask[..., None, None]
        gram = np.einsum("vnjri,vnjrk->njik", weighted, weighted)
        vectors = np.linalg.eigh(gram)[1][..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            points = vectors[..., :3] / vectors[..., 3:]
        points[mask.sum(0) < 2] = np.nan
        return points

    def errors(points: Float64[ndarray, "n j 3"]) -> Float64[ndarray, "v n j"]:
        flat = np.nan_to_num(points).reshape(-1, 3)
        out = []
        for v, p in enumerate(phones):
            uvd = cameras[p].project(flat)
            error = np.linalg.norm(uvd[:, :2] - xy[v].reshape(-1, 2), axis=1)
            out.append(np.where(uvd[:, 2] > 0.0, error, np.inf).reshape(points.shape[:2]))
        return np.where(np.isfinite(points).all(-1)[None], np.stack(out), np.inf)

    mask = use
    for _ in range(2):
        mask = use & (errors(solve(mask)) <= gate_px)
    points = solve(mask)
    rays = points[None] - centres[:, None, None]
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    cosines = np.einsum("anjk,bnjk->abnj", rays, rays)
    pair = mask[:, None] & mask[None] & ~np.eye(len(phones), dtype=bool)[..., None, None]
    widest = np.where(pair, cosines, 1.0).min(axis=(0, 1))
    points[~(widest < np.cos(np.radians(MIN_BODY_RAY_DEG)))] = np.nan
    return points


def body_correspondences(camera: Camera, body: BodyTrack, points: Float64[ndarray, "n j 3"]) -> Correspondences:
    """A phone's body detections paired with the triangulated body, within BODY_GATE_PX, thinned to MAX_BODY_PAIRS."""
    keep = (body.scores >= MIN_BODY_SCORE) & np.isfinite(body.xy).all(-1) & np.isfinite(points).all(-1)
    data = Correspondences(points[keep], body.xy[keep], body.scores[keep])
    if not len(data.world):
        return data
    near = camera.residuals(data) <= BODY_GATE_PX
    return _thin(Correspondences(data.world[near], data.image[near], data.weights[near]), MAX_BODY_PAIRS)


def body_bundle(cameras: Mapping[str, Camera], faces: Mapping[str, FaceTrack], hands: Mapping[str, Correspondences],
                bodies: Mapping[str, BodyTrack], track: HeadTrack, offsets: Float64[ndarray, "5 3"], rounds: int = BODY_ROUNDS) -> dict[str, Camera]:
    """Refit every pose to its face and hand pairs plus the wearer's body as the other phones see it.

    Face and hands anchor each phone to the head track on their own, so phones pose well near the head and hands but
    can disagree far below them. Each round triangulates the body from the current poses and refits all poses to
    face, hand and body pairs (offsets fixed). Phones without a body track keep only their own pairs.
    """
    fitted = dict(cameras)
    empty = Correspondences(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0))
    for _ in range(rounds):
        points = triangulate_bodies(fitted, bodies)
        pairs = {}
        for p, camera in fitted.items():
            own = hands.get(p, empty)
            extra = body_correspondences(camera, bodies[p], points) if p in bodies and len(points) else empty
            pairs[p] = Correspondences(np.r_[own.world, extra.world], np.r_[own.image, extra.image], np.r_[own.weights, extra.weights])
        fitted, _ = joint_bundle(fitted, faces, pairs, track, offsets, fit_offsets=False)
    return fitted


DEFAULT_OFFSETS: Float64[ndarray, "5 3"] = np.array([
    [0.0311, -0.0064, -0.0646],
    [0.0042, -0.0295, -0.0906],
    [0.0613, -0.0232, -0.0938],
    [-0.0587, -0.0056, -0.1536],
    [0.1018, 0.0148, -0.1652],
])
"""RTMW face points on a Quest 3S wearer, in the Quest left-camera frame (x right, y down, z forward), metres.

Measured on selfcap__2025-12-03__d31bf1da__ep001 from its four label-refined phone cameras (headset labels within
1–8 px): each joint triangulated per frame, moved into the head frame, median over 3,640 frames. Face points of a
headset wearer are guesses the detector makes around the visor, so offsets fitted freely per session absorb camera
error; this fixed template is what pins the cameras. Another ego device or detector needs its own template.
"""
WEARER_GATE_PX: float = 150.0
"""A person is the wearer only if its face lies this close to the projected head."""
ACCEPT_FACE_PX: float = 25.0
"""A head-anchored phone needs its median face residual under this to replace or create a pose."""
CLOCK_SUSPECT_S: float = 0.3
"""A clock scan whose best shift is at least this far from zero, ..."""
CLOCK_SUSPECT_GAIN: float = 1.5
"""... and fits the face this many times better, marks the phone's clock as wrong."""
HAND_GATE_PX: float = 50.0
"""Hand pairs farther than this from the projected Quest hands are dropped (a phone that never sees the hands still
reports some, from the arms or the box edge)."""


@dataclass(frozen=True, slots=True)
class PhoneReport:
    """Provenance of one phone's head-anchored solve."""

    method: str
    """'hands' (unchanged), 'head' (new pose from the head track), 'hands+head' (joint refit) or 'unposed'."""
    face_pairs: int
    face_median_px: float | None
    hand_pairs: int
    hand_median_px: float | None
    centre_shift_m: float | None
    """Camera-centre change against the hand-only pose, when both exist."""
    hands_only_face_px: float | None = None
    """Face residual of the hands-only pose, for comparison."""
    clock_scan_best_s: float | None = None
    """Clock shift (s) at which the head-only re-solve fits the face best; near zero when the phone's clock is right."""
    clock_scan_gain: float | None = None
    """Face residual at zero shift divided by the best one; ~1 for a right clock."""
    clock_correction_ms: float | None = None
    """New phone offset (ms) when the head track found the clock missing or wrong."""


def select_person(boxes: Float64[ndarray, "n p 4"], box_scores: Float64[ndarray, "n p"], xy: Float64[ndarray, "n p 133 2"],
                  scores: Float64[ndarray, "n p 133"], expected: Float64[ndarray, "n 2"] | None) -> ndarray:
    """Per frame, the candidate that is the wearer; -1 when none qualifies.

    Without a camera, the most confident person whose face is seen; with one, the person whose face centroid lies
    nearest the projected head, within WEARER_GATE_PX.
    """
    face_ok = scores[..., list(FACE)] >= MIN_FACE_SCORE
    seen = face_ok.sum(-1) >= 2
    with np.errstate(invalid="ignore"):
        centroid = np.nansum(np.where(face_ok[..., None], xy[..., list(FACE), :], 0.0), axis=2) / np.maximum(face_ok.sum(-1), 1)[..., None]
    if expected is None:
        ranked = np.where(seen, np.nan_to_num(box_scores, nan=-1.0), -np.inf)
        best = np.argmax(ranked, axis=1)
        return np.where(np.isfinite(np.take_along_axis(ranked, best[:, None], 1)[:, 0]), best, -1)
    distance = np.where(seen, np.linalg.norm(centroid - expected[:, None], axis=-1), np.inf)
    distance[~np.isfinite(expected).all(-1)] = np.inf
    best = np.argmin(distance, axis=1)
    return np.where(np.take_along_axis(distance, best[:, None], 1)[:, 0] < WEARER_GATE_PX, best, -1)


def chosen(values: ndarray, choice: ndarray, fill: float) -> ndarray:
    """Rows of ``values`` [n, p, ...] at ``choice`` [n]; ``fill`` where choice is -1."""
    picked = np.take_along_axis(values, np.maximum(choice, 0).reshape(-1, *([1] * (values.ndim - 1))), axis=1)[:, 0]
    picked[choice < 0] = fill
    return picked


def residual_summary(camera: Camera, data: Correspondences) -> tuple[int, float | None]:
    """Pair count and median pixel residual."""
    if not len(data.world):
        return 0, None
    residual = camera.residuals(data)
    finite = residual[np.isfinite(residual)]
    return len(data.world), float(np.median(finite)) if len(finite) else None


def gate_hands(camera: Camera, data: Correspondences) -> Correspondences:
    """Hand pairs within HAND_GATE_PX of the camera's projection of the Quest hands."""
    if not len(data.world):
        return data
    keep = camera.residuals(data) <= HAND_GATE_PX
    return Correspondences(data.world[keep], data.image[keep], data.weights[keep])


def _thin(data: Correspondences, limit: int = 1200) -> Correspondences:
    """At most ``limit`` evenly spaced pairs, to keep clock scans cheap."""
    if len(data.world) <= limit:
        return data
    keep = np.linspace(0, len(data.world) - 1, limit).round().astype(int)
    return Correspondences(data.world[keep], data.image[keep], data.weights[keep])


CLOCK_SCAN_SHIFTS_S: Float64[ndarray, "k"] = np.arange(-3.0, 3.0001, 0.2)
"""Coarse clock shifts the scan tries: +/-3 s in 0.2 s steps."""


def clock_scan(face: FaceTrack, track: HeadTrack, offsets: Float64[ndarray, "5 3"], seed: Camera | None,
               shifts: Float64[ndarray, "k"] = CLOCK_SCAN_SHIFTS_S, focal: float | None = None) -> tuple[float | None, float | None]:
    """Re-solve the pose from the face alone at each clock shift; the best shift and its gain over zero shift.

    With no seed, every shift solves from scratch (EPnP RANSAC), which finds a phone's clock without any prior.
    """
    best, at_zero = (None, np.inf), None
    for shift in shifts:
        moved = FaceTrack(face.times_s + shift, face.xy, face.scores)
        data = _thin(face_correspondences(moved, track, offsets))
        camera = solve_from_head(data, focal if seed is None else float(seed.parameters[6]), seed=seed)
        if camera is None:
            continue
        _, median = residual_summary(camera, data)
        if median is None:
            continue
        if abs(shift) < 1e-9:
            at_zero = median
        if median < best[1]:
            best = (float(shift), median)
    if best[0] is None:
        return None, None
    return best[0], None if at_zero is None else float(at_zero / max(best[1], 1e-9))


def find_clock(face: FaceTrack, track: HeadTrack, offsets: Float64[ndarray, "5 3"], focal: float, half_range_s: float = 15.0) -> float | None:
    """A phone's clock shift from its face track alone: a coarse scan from scratch, then a fine seeded one."""
    coarse, _ = clock_scan(face, track, offsets, None, np.arange(-half_range_s, half_range_s + 1e-9, 0.25), focal=focal)
    if coarse is None:
        return None
    moved = FaceTrack(face.times_s + coarse, face.xy, face.scores)
    seed = solve_from_head(face_correspondences(moved, track, offsets), focal)
    if seed is None:
        return coarse
    fine, _ = clock_scan(face, track, offsets, seed, np.arange(coarse - 0.4, coarse + 0.4001, 0.04))
    return coarse if fine is None else fine


UPSIDE_DOWN_COS: float = 0.5
"""A phone whose image "down" axis is within 60 degrees of world up is upside down."""


def world_up(track: HeadTrack) -> Float64[ndarray, "3"]:
    """World up, opposite the head camera's mean "down" axis: a worn headset is upright on average (its pitch tilts this
    estimate by tens of degrees at most, far inside the upside-down test below)."""
    down = track.rotations[:, :, 1].mean(axis=0)
    return -down / np.linalg.norm(down)


def upright(camera: Camera, up: Float64[ndarray, "3"]) -> bool:
    """A mounted phone is never upside down. Five face points seen from a few metres also fit a mirrored pose with the
    image turned over; this rejects it."""
    return float(camera.world_T_cam[:3, 1] @ up) < UPSIDE_DOWN_COS
