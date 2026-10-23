import numpy as np
from conftest import project_points, ring_rig
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from exo_calib.ops.boxes import (
    box_iou,
    boxes_from_projected_keypoints,
    boxes_needing_detection,
    extrapolate_keypoints,
    nearest_candidates,
    skeleton_centre,
    vote_person,
)


def test_boxes_from_projected_keypoints_filters_confidence_and_pads_about_center() -> None:
    points_xyz: Float64[ndarray, "4 3"] = np.array(
        [[-1.0, -0.5, 5.0], [1.0, 0.5, 5.0], [10.0, 10.0, 5.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )
    conf: Float64[ndarray, "4"] = np.array([0.9, 0.8, 0.1, 1.0], dtype=np.float64)
    camera_intrinsics: Float64[ndarray, "3 3"] = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    camera_T_world: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)

    box_xyxy: Float64[ndarray, "4"] = boxes_from_projected_keypoints(
        points_xyz,
        conf,
        camera_intrinsics,
        camera_T_world,
        image_wh=(100, 80),
        pad=1.25,
        min_joints=2,
        min_confidence=0.5,
    )

    np.testing.assert_allclose(box_xyxy, np.array([25.0, 27.5, 75.0, 52.5], dtype=np.float64))


def test_boxes_from_projected_keypoints_clips_padding_and_rejects_insufficient_joints() -> None:
    camera_intrinsics: Float64[ndarray, "3 3"] = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    camera_T_world: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    clipped_points_xyz: Float64[ndarray, "2 3"] = np.array([[-2.25, -1.75, 5.0], [-1.75, -1.25, 5.0]], dtype=np.float64)

    clipped_xyxy: Float64[ndarray, "4"] = boxes_from_projected_keypoints(
        clipped_points_xyz,
        np.ones(2, dtype=np.float64),
        camera_intrinsics,
        camera_T_world,
        image_wh=(100, 80),
        pad=2.0,
        min_joints=2,
        min_confidence=0.15,
    )
    missing_xyxy: Float64[ndarray, "4"] = boxes_from_projected_keypoints(
        np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 5.0]], dtype=np.float64),
        np.ones(2, dtype=np.float64),
        camera_intrinsics,
        camera_T_world,
        image_wh=(100, 80),
        min_joints=2,
        pad=1.25,
        min_confidence=0.15,
    )

    np.testing.assert_allclose(clipped_xyxy, np.array([0.0, 0.0, 20.0, 20.0], dtype=np.float64))
    assert np.isnan(missing_xyxy).all()


def test_boxes_needing_detection_flags_missing_and_image_boundary_boxes() -> None:
    boxes: Float64[ndarray, "5 4"] = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, 10.0, 80.0, 70.0],
            [10.0, 0.0, 80.0, 70.0],
            [10.0, 10.0, 100.0, 80.0],
            [10.0, 10.0, 80.0, 70.0],
        ],
        dtype=np.float64,
    )

    needs_detection: Bool[ndarray, " n"] = boxes_needing_detection(boxes, image_wh=(100, 80))

    np.testing.assert_array_equal(needs_detection, np.array([True, True, True, True, False]))


def test_extrapolate_keypoints_uses_velocity_where_available_and_latest_elsewhere() -> None:
    latest_points_xyz: Float64[ndarray, "3 3"] = np.array(
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [np.nan, np.nan, np.nan]], dtype=np.float64
    )
    previous_points_xyz: Float64[ndarray, "3 3"] = np.array(
        [[1.0, 1.0, 1.0], [np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]], dtype=np.float64
    )

    predicted_points_xyz: Float64[ndarray, "3 3"] = extrapolate_keypoints(latest_points_xyz, previous_points_xyz, step_ratio=2.0)

    expected_points_xyz: Float64[ndarray, "3 3"] = np.array(
        [[4.0, 7.0, 10.0], [5.0, 6.0, 7.0], [np.nan, np.nan, np.nan]], dtype=np.float64
    )
    np.testing.assert_allclose(predicted_points_xyz, expected_points_xyz, equal_nan=True)


RING_K: Float64[ndarray, "4 3 3"] = np.repeat(np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])[None], 4, axis=0)
RING_T: Float64[ndarray, "4 4 4"] = ring_rig(4, 4.0)
PERSON_A: Float64[ndarray, "3"] = np.array([0.5, 0.2, 0.0])
PERSON_B: Float64[ndarray, "3"] = np.array([-0.8, -0.6, 0.0])  # same height as A and as the cameras: every epipolar line is level


def person_box(center_xyz: Float64[ndarray, "3"], view: int) -> Float64[ndarray, "4"]:
    """The XYXY box of a 0.5 x 0.5 x 1.7 m person centred at ``center_xyz``."""
    corners: Float64[ndarray, "8 3"] = center_xyz + np.array([[x, y, z] for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.85, 0.85)])
    uv: Float64[ndarray, "8 2"] = project_points(corners, RING_K[view : view + 1], RING_T[view : view + 1])[0]
    return np.concatenate((uv.min(axis=0), uv.max(axis=0)))


PEOPLE: dict[str, Float64[ndarray, "3"]] = {"A": PERSON_A, "B": PERSON_B}


def who_each_view_takes(seen: dict[int, list[tuple[str, float]]], person_xyz: Float64[ndarray, "3"] | None) -> list[str | None]:
    """The person each view takes (None: nobody) among the detections ``seen`` lists per view (person, score): the candidate
    nearest ``person_xyz``, or nearest the person ``vote_person`` picks when it is None."""
    detections: list[tuple[int, str, float]] = [(view, person, score) for view, people in seen.items() for person, score in people]
    boxes: Float64[ndarray, "n 4"] = np.array([person_box(PEOPLE[person], view) for view, person, _ in detections]).reshape(-1, 4)
    scores: Float64[ndarray, " n"] = np.array([score for *_, score in detections])
    view_of: Int64[ndarray, " n"] = np.array([view for view, *_ in detections], dtype=np.int64)
    projection: Float64[ndarray, "4 3 4"] = RING_K @ RING_T[:, :3]
    centre = vote_person(boxes, scores, view_of, projection) if person_xyz is None else person_xyz
    if centre is None:
        return [None] * 4
    rows = nearest_candidates(centre[None], boxes, view_of, projection)[0][0]
    return [detections[row][1] if row >= 0 else None for row in rows]


def test_without_a_track_the_person_most_views_agree_on_wins_over_a_higher_score() -> None:
    """B out-scores A where both are seen, but only two views see B; A's box centres agree in all four views."""
    seen = {v: [("A", 0.6), *([("B", 0.95)] if v in (0, 2) else [])] for v in range(4)}

    assert who_each_view_takes(seen, person_xyz=None) == ["A", "A", "A", "A"]


def test_with_a_track_each_view_takes_the_candidate_nearest_the_tracked_person() -> None:
    """View 1 sees both (B scores higher) and takes A; view 2 sees only B and takes B (refine rejects it); view 3 sees nobody."""
    seen = {0: [("A", 0.6)], 1: [("B", 0.95), ("A", 0.6)], 2: [("B", 0.95)], 3: []}

    assert who_each_view_takes(seen, person_xyz=PERSON_A) == ["A", "A", "B", None]


def test_a_badly_calibrated_view_still_takes_the_person_nearer_its_projection() -> None:
    """The rig is what is being calibrated: with the anchor 0.4 m off, view 0's projection misses A's box, yet A is still the
    nearer of the two people there, so view 0 takes A instead of nothing."""
    seen = {v: [("B", 0.95), ("A", 0.6)] for v in range(4)}

    assert who_each_view_takes(seen, person_xyz=PERSON_A + np.array([0.0, 0.4, 0.0]))[0] == "A"


def test_without_detections_no_view_takes_anyone() -> None:
    assert who_each_view_takes({v: [] for v in range(4)}, person_xyz=None) == [None] * 4


def test_the_skeleton_centre_is_the_median_of_its_confident_finite_joints() -> None:
    points: Float64[ndarray, "4 3"] = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [np.nan, 0.0, 0.0], [9.0, 9.0, 9.0]])
    conf: Float64[ndarray, " 4"] = np.array([0.9, 0.9, 0.9, 0.1])

    np.testing.assert_allclose(skeleton_centre(points, conf, 0.5), [1.0, 1.0, 1.0])  # type: ignore[arg-type]
    assert skeleton_centre(points, np.zeros(4), 0.5) is None


def test_box_iou_row_by_row_and_nan_where_a_box_is_missing_or_both_are_empty() -> None:
    first: Float64[ndarray, "5 4"] = np.array([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0], [np.nan] * 4, [1.0, 1.0, 1.0, 1.0]])
    second: Float64[ndarray, "5 4"] = np.array([[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    np.testing.assert_allclose(box_iou(first, second), [1.0 / 7.0, 0.0, 1.0, np.nan, np.nan])  # overlap 1 of 4 + 4 - 1
