"""Exo cameras from the head track: synthetic scenes with a known camera, head path and face offsets."""

import numpy as np
from dataforge.selfcap_schema import rdf_head
from scipy.spatial.transform import Rotation

from selfcap_curate.exo_calibration import Camera
from selfcap_curate.head_anchor import FaceTrack, HeadTrack, estimate_offsets, face_correspondences, joint_bundle, solve_from_head
from selfcap_curate.head_calibration import head_track
from selfcap_curate.records import CatalogHands

OFFSETS = np.array([[0.03, 0.04, -0.06], [0.0, 0.0, -0.04], [0.063, 0.0, -0.04], [-0.04, 0.02, 0.06], [0.10, 0.02, 0.06]])


def look_at(centre: np.ndarray, target: np.ndarray, focal: float = 1400.0) -> Camera:
    """An OpenCV camera at ``centre`` aimed at ``target`` with world +y up."""
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    world_R_cam = np.c_[right, down, forward]
    cam_R_world = world_R_cam.T
    return Camera(np.r_[Rotation.from_matrix(cam_R_world).as_rotvec(), -cam_R_world @ centre, focal, 960.0, 540.0, 0.0, 0.0])


def walk(n: int = 1600, seed: int = 3) -> HeadTrack:
    """A standing person over 80 s at 20 Hz: the head wanders ~0.5 m and turns up to ~40 degrees."""
    rng = np.random.default_rng(seed)
    times = np.arange(n) / 20.0
    positions = np.array([0.0, 1.6, 0.0]) + np.cumsum(rng.normal(0.0, 0.0045, (n, 3)), axis=0).clip(-0.3, 0.3)
    angles = np.c_[np.sin(times / 7.0) * 0.5, np.sin(times / 5.0) * 0.7, np.sin(times / 11.0) * 0.2]
    return HeadTrack(times, Rotation.from_euler("yxz", angles).as_matrix(), positions)


def observe(camera: Camera, track: HeadTrack, noise: float, outliers: float, seed: int = 5) -> FaceTrack:
    """The face seen at 4 Hz, with pixel noise and gross outliers."""
    rng = np.random.default_rng(seed)
    rows = slice(None, None, 5)
    world = np.einsum("nij,kj->nki", track.rotations[rows], OFFSETS) + track.positions[rows, None]
    n = len(world)
    uv = camera.project(world.reshape(-1, 3))[:, :2].reshape(n, 5, 2) + rng.normal(0.0, noise, (n, 5, 2))
    bad = rng.random((n, 5)) < outliers
    uv[bad] += rng.uniform(-200.0, 200.0, (int(bad.sum()), 2))
    return FaceTrack(track.times_s[rows], uv, np.full((n, 5), 0.8))


def errors(found: Camera, truth: Camera) -> tuple[float, float]:
    """Camera-centre distance (m) and rotation angle (degrees)."""
    centre = found.world_T_cam[:3, 3] - truth.world_T_cam[:3, 3]
    angle = (Rotation.from_rotvec(found.parameters[:3]) * Rotation.from_rotvec(truth.parameters[:3]).inv()).magnitude()
    return float(np.linalg.norm(centre)), float(np.degrees(angle))


def test_a_phone_that_only_sees_the_head_is_solved_from_the_head_track() -> None:
    track = walk()
    truth = look_at(np.array([2.5, 2.2, 1.8]), track.positions.mean(0))
    face = observe(truth, track, noise=2.0, outliers=0.1)
    camera = solve_from_head(face_correspondences(face, track, OFFSETS), focal=1400.0)
    assert camera is not None
    distance, angle = errors(camera, truth)
    assert distance < 0.03 and angle < 0.5


def test_face_offsets_come_from_two_calibrated_phones() -> None:
    track = walk()
    cameras = {"a": look_at(np.array([2.5, 2.2, 1.8]), np.array([0.0, 1.6, 0.0])), "b": look_at(np.array([-2.0, 2.0, 2.5]), np.array([0.0, 1.6, 0.0]))}
    faces = {phone: observe(camera, track, noise=1.5, outliers=0.05, seed=i) for i, (phone, camera) in enumerate(cameras.items())}
    offsets = estimate_offsets(cameras, faces, track)
    assert offsets is not None
    assert np.nanmax(np.linalg.norm(offsets - OFFSETS, axis=1)) < 0.01


def test_the_joint_fit_refines_rough_offsets_and_a_rough_camera() -> None:
    track = walk()
    cameras = {"a": look_at(np.array([2.5, 2.2, 1.8]), np.array([0.0, 1.6, 0.0])), "b": look_at(np.array([-2.0, 2.0, 2.5]), np.array([0.0, 1.6, 0.0]))}
    faces = {phone: observe(camera, track, noise=1.5, outliers=0.05, seed=i) for i, (phone, camera) in enumerate(cameras.items())}
    rough_offsets = OFFSETS + np.random.default_rng(9).normal(0.0, 0.02, OFFSETS.shape)
    start = {p: solve_from_head(face_correspondences(faces[p], track, rough_offsets), focal=1400.0) for p in cameras}
    assert all(c is not None for c in start.values())
    fitted, offsets = joint_bundle({p: c for p, c in start.items() if c is not None}, faces, {}, track, rough_offsets)
    for phone, camera in fitted.items():
        distance, angle = errors(camera, cameras[phone])
        assert distance < 0.02 and angle < 0.3
    assert np.max(np.linalg.norm(offsets - OFFSETS, axis=1)) < 0.01


def test_a_head_track_stored_with_y_up_and_z_back_is_turned_back_to_rdf() -> None:
    """Some captures store the eye orientation as x right, y up, z back: the camera's down axis then points up the world."""
    level = look_at(np.zeros(3), np.array([0.0, -0.3, 1.0])).world_T_cam[:3, :3]  # RDF, looking ahead and a little down
    n = 50

    def track(rotation: np.ndarray) -> HeadTrack:
        quaternions = Rotation.from_matrix(np.repeat(rotation[None], n, axis=0)).as_quat()
        axes = rdf_head(quaternions)
        return head_track(CatalogHands(np.zeros(0), np.zeros((0, 133, 3)), np.arange(n) / 20.0, np.tile([0.0, 1.6, 0.0], (n, 1)), axes.quaternions, axes.convention), 0.0)

    stored_rub = track(level @ np.diag([1.0, -1.0, -1.0]))
    stored_rdf = track(level)
    assert np.allclose(stored_rub.rotations, level, atol=1e-9) and stored_rub.convention == "RUB"
    assert np.allclose(stored_rdf.rotations, level, atol=1e-9) and stored_rdf.convention == "RDF"


def test_a_hands_only_pose_the_head_fit_cannot_check_is_flagged_for_the_episode() -> None:
    """Keypoints keeps calib_status=ok cameras only: an unchecked pose must not feed the triangulation."""
    from selfcap_curate.exo_calibration import EpisodeCalibration, PhoneCalibration
    from selfcap_curate.head_calibration import unverified

    camera = look_at(np.array([2.0, 1.5, 0.0]), np.array([0.0, 1.2, 0.0]))
    phone = PhoneCalibration(True, camera, 9.0, 20.0, 240, {"ep001": EpisodeCalibration(240, 9.0, True, "session")})
    flagged = unverified(phone, "ep001")
    assert not flagged.ok and flagged.camera is camera
    assert not flagged.episodes["ep001"].ok and flagged.episodes["ep001"].scope == "flagged"
    assert unverified(PhoneCalibration(True, camera), "ep002").episodes["ep002"].scope == "flagged"


def test_an_upside_down_phone_pose_is_not_upright_in_the_head_tracks_world() -> None:
    """Five face points seen from afar also fit a mirrored, upside-down pose; a mounted phone is never upside down."""
    from selfcap_curate.head_anchor import upright, world_up

    level = look_at(np.zeros(3), np.array([0.0, -0.5, 1.0])).world_T_cam[:3, :3]  # the wearer looks ahead and down
    track = HeadTrack(np.arange(20) / 20.0, np.repeat(level[None], 20, axis=0), np.tile([0.0, 1.6, 0.0], (20, 1)))
    up = world_up(track)
    assert up[1] > 0.8
    phone = look_at(np.array([2.0, 2.0, 1.0]), np.array([0.0, 1.4, 0.0]))
    flipped = Camera(np.r_[(Rotation.from_rotvec(phone.parameters[:3]).inv() * Rotation.from_euler("z", 180, degrees=True)).inv().as_rotvec(),
                           phone.parameters[3:]])
    assert upright(phone, up) and not upright(flipped, up)


# The wearer's body below the head (world offsets from the head position): shoulders, elbows, wrists, hips, knees, ankles.
BODY_OFFSETS = np.array([[0.18, -0.25, 0.0], [-0.18, -0.25, 0.0], [0.25, -0.5, 0.05], [-0.25, -0.5, 0.05], [0.25, -0.75, 0.15], [-0.25, -0.75, 0.15],
                         [0.1, -0.75, 0.0], [-0.1, -0.75, 0.0], [0.1, -1.15, 0.02], [-0.1, -1.15, 0.02], [0.1, -1.55, 0.0], [-0.1, -1.55, 0.0]])


def observe_body(camera: Camera, track: HeadTrack, noise: float, seed: int = 7):
    """The body seen at 4 Hz on the face grid, with pixel noise."""
    from selfcap_curate.head_anchor import BodyTrack

    rng = np.random.default_rng(seed)
    rows = slice(None, None, 5)
    world = track.positions[rows, None] + BODY_OFFSETS[None]
    n = len(world)
    uv = camera.project(world.reshape(-1, 3))[:, :2].reshape(n, len(BODY_OFFSETS), 2) + rng.normal(0.0, noise, (n, len(BODY_OFFSETS), 2))
    return BodyTrack(track.times_s[rows], uv, np.full((n, len(BODY_OFFSETS)), 0.9))


def leg_disagreement(cameras: dict[str, Camera], bodies) -> float:
    """Median largest reprojection error of the legs (knees, ankles) triangulated from every view."""
    from selfcap_curate.head_anchor import triangulate_bodies

    points = triangulate_bodies(cameras, bodies, gate_px=1e6)
    errors = [np.linalg.norm(cameras[p].project(points[:, 8:].reshape(-1, 3))[:, :2] - bodies[p].xy[:, 8:].reshape(-1, 2), axis=1) for p in cameras]
    return float(np.nanmedian(np.max(errors, axis=0)))


SCENE = {"a": (np.array([2.5, 2.2, 1.8]), np.array([0.0, 1.0, 0.0])), "b": (np.array([-2.0, 2.0, 2.5]), np.array([0.0, 1.0, 0.0])),
         "c": (np.array([0.5, 1.9, -3.0]), np.array([0.0, 1.0, 0.0]))}


def test_body_joints_are_triangulated_from_the_views_that_agree() -> None:
    from selfcap_curate.head_anchor import BodyTrack, triangulate_bodies

    track = walk()
    cameras = {p: look_at(*scene) for p, scene in SCENE.items()}
    bodies = {p: observe_body(c, track, noise=0.0, seed=i) for i, (p, c) in enumerate(cameras.items())}
    bad = bodies["c"].xy.copy()
    bad[:, 0] += 80.0  # one view puts the left shoulder 80 px off: the other two still agree
    alone = bodies["b"].scores.copy()
    alone[:, 11] = 0.0  # right ankle seen by b only after a and c lose it
    bodies = {"a": BodyTrack(bodies["a"].times_s, bodies["a"].xy, np.where(np.arange(12) == 11, 0.0, bodies["a"].scores)),
              "b": BodyTrack(bodies["b"].times_s, bodies["b"].xy, np.where(np.arange(12) == 11, 0.9, alone)),
              "c": BodyTrack(bodies["c"].times_s, bad, np.where(np.arange(12) == 11, 0.0, bodies["c"].scores))}
    points = triangulate_bodies(cameras, bodies)
    truth = track.positions[::5, None] + BODY_OFFSETS[None]
    assert np.nanmax(np.linalg.norm(points[:, :11] - truth[:, :11], axis=-1)) < 1e-3
    assert np.isnan(points[:, 11]).all()


def test_body_joints_seen_only_by_two_phones_facing_each_other_are_missing() -> None:
    """Joints near the line between two facing phones: their rays are under 5 degrees apart as lines, so depth is free;
    a third phone from the side fixes them."""
    from selfcap_curate.head_anchor import BodyTrack, triangulate_bodies

    rng = np.random.default_rng(11)
    truth = np.array([0.0, 1.0, 0.0]) + rng.uniform([-0.05, -0.05, -0.3], [0.05, 0.05, 0.3], (20, 12, 3))
    cameras = {p: look_at(np.array(centre), np.array([0.0, 1.0, 0.0])) for p, centre in
               (("front", [0.0, 1.0, 3.0]), ("back", [0.0, 1.0, -3.0]), ("side", [3.0, 1.0, 0.0]))}
    bodies = {p: BodyTrack(np.arange(20) / 4.0, c.project(truth.reshape(-1, 3))[:, :2].reshape(20, 12, 2), np.full((20, 12), 0.9))
              for p, c in cameras.items()}
    facing = {p: cameras[p] for p in ("front", "back")}
    assert np.isnan(triangulate_bodies(facing, bodies)).all()
    np.testing.assert_allclose(triangulate_bodies(cameras, bodies), truth, atol=1e-6)


def test_the_body_seen_from_several_phones_corrects_poses_a_wrong_face_template_leaves_off() -> None:
    """Face pairs alone pose each phone separately: a template 1-2 cm off turns each phone a little, and the legs, far
    below the head, disagree between phones. The body seen by every phone pulls them back into agreement."""
    from selfcap_curate.head_anchor import body_bundle

    track = walk()
    truth = {p: look_at(*scene) for p, scene in SCENE.items()}
    faces = {p: observe(c, track, noise=1.5, outliers=0.05, seed=i) for i, (p, c) in enumerate(truth.items())}
    bodies = {p: observe_body(c, track, noise=1.5, seed=10 + i) for i, (p, c) in enumerate(truth.items())}
    wrong = OFFSETS + np.random.default_rng(4).normal(0.0, 0.015, OFFSETS.shape)
    start = {p: solve_from_head(face_correspondences(faces[p], track, wrong), focal=1400.0) for p in truth}
    assert all(c is not None for c in start.values())
    face_only, _ = joint_bundle({p: c for p, c in start.items() if c is not None}, faces, {}, track, wrong, fit_offsets=False)
    refined = body_bundle(face_only, faces, {}, bodies, track, wrong)
    before, after = leg_disagreement(face_only, bodies), leg_disagreement(refined, bodies)
    assert before > 6.0 and after < before / 2
    # Agreement alone does not fix the phones' common orientation; the face (and in captures the Quest hands) still do,
    # so the refined poses must keep fitting the face.
    from selfcap_curate.head_anchor import residual_summary

    for p in truth:
        face_before = residual_summary(face_only[p], face_correspondences(faces[p], track, wrong))[1]
        face_after = residual_summary(refined[p], face_correspondences(faces[p], track, wrong))[1]
        assert face_before is not None and face_after is not None and face_after < face_before + 2.0



def test_fixed_phones_are_posed_from_a_few_frames_spread_over_the_session(fixture_sync) -> None:
    # Fixed phones need no dense pass: align's few detections per phone, spread over the session and read on each
    # phone's own clock, pin every pose.
    from selfcap_curate.exo_calibration import PhoneCalibration, SessionCalibration
    from selfcap_curate.head_anchor import DEFAULT_OFFSETS, FACE
    from selfcap_curate.head_calibration import calibrate_with_head
    from selfcap_curate.records import CatalogHands, PhoneDetections, SelfcapLayout, SessionEpisode

    walked = walk()  # turned into an upright RDF head (camera y down), as worn
    level = look_at(np.zeros(3), np.array([0.0, 0.0, 1.0])).world_T_cam[:3, :3]
    track = HeadTrack(walked.times_s, walked.rotations @ level, walked.positions)
    truth = {"A": look_at(np.array([2.0, 1.5, 2.0]), np.array([0.0, 1.4, 0.0])), "B": look_at(np.array([-2.0, 1.2, 1.5]), np.array([0.0, 1.5, 0.0]))}
    clock_ms = {"A": 100.0, "B": 60.0}
    detections: dict[str, PhoneDetections] = {}
    rows = np.linspace(40, len(track.times_s) - 40, 48).round().astype(int)  # align detects the same frames on every phone
    for phone, camera in truth.items():
        world = np.einsum("nij,kj->nki", track.rotations[rows], DEFAULT_OFFSETS) + track.positions[rows, None]
        xy, scores = np.full((48, 133, 2), np.nan), np.zeros((48, 133))
        xy[:, list(FACE)] = camera.project(world.reshape(-1, 3))[:, :2].reshape(48, 5, 2)
        scores[:, list(FACE)] = 0.9
        detections[phone] = PhoneDetections(track.times_s[rows] - clock_ms[phone] / 1000.0, xy, scores)
    episode = SessionEpisode(
        fixture_sync(offsets_ms={f"exo/{phone}": ms for phone, ms in clock_ms.items()}),
        CatalogHands(np.empty(0), np.empty((0, 133, 3)), np.empty(0), np.empty((0, 3)), np.empty((0, 4))),
        np.empty(0), np.empty((0, 3)), {}, SelfcapLayout(2, 3, ("A", "B")), detections,
    )
    head = calibrate_with_head(SessionCalibration("s", {p: PhoneCalibration(False) for p in truth}), episode, track, {p: 1400.0 for p in truth}, clock_ms)
    for phone, camera in truth.items():
        centre, angle = errors(head.exo.phones[phone].camera, camera)
        assert centre < 0.03 and angle < 1.0, (phone, centre, angle)
