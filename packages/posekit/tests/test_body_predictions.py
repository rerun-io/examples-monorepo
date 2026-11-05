"""Public boundary tests for whole-body 3D predictions and the MHR-70 skeleton (CPU-only)."""

import pytest
import torch

from posekit.predictions import BodyPoses3d, Keypoints3d
from posekit.skeletons import MHR_70, SKELETONS


def body_rows(people: int, boxes: int) -> BodyPoses3d:
    return BodyPoses3d(
        keypoints=Keypoints3d.empty(MHR_70, "cpu", root_indices=(9, 10), with_xyz_camera=True)
        if people == 0
        else Keypoints3d(
            torch.zeros(people, 70, 2), torch.zeros(people, 70), torch.ones(people, 70), torch.zeros(people, dtype=torch.int64), MHR_70, (9, 10), torch.zeros(people, 70, 3)
        ),
        boxes=torch.zeros(boxes, 4),
        vertices=torch.zeros(people, 18439, 3),
        faces=torch.zeros(36874, 3, dtype=torch.int64),
    )


def test_mhr70_matches_upstream() -> None:
    """posekit's MHR-70 names and links are upstream's ``pose_info`` (sam-3d-body-src), in its order."""
    mhr70 = pytest.importorskip("sam_3d_body.metadata.mhr70", reason="SAM 3D Body needs the posekit-body Pixi environment")
    keypoints: dict[int, dict[str, object]] = mhr70.pose_info["keypoint_info"]
    index_of: dict[object, int] = {info["name"]: index for index, info in keypoints.items()}
    assert MHR_70.keypoint_names == tuple(keypoints[index]["name"] for index in range(len(keypoints)))
    assert MHR_70.links == tuple((index_of[a], index_of[b]) for a, b in (info["link"] for info in mhr70.pose_info["skeleton_info"].values()))


def test_mhr70_names_body_hands_and_links() -> None:
    """70 named joints (both 21-joint hands included) whose links stay inside the skeleton."""
    assert MHR_70.num_keypoints == 70
    assert SKELETONS["mhr_70"] is MHR_70
    assert MHR_70.keypoint_names[9:11] == ("left_hip", "right_hip")
    assert (MHR_70.keypoint_names[21], MHR_70.keypoint_names[41], MHR_70.keypoint_names[42], MHR_70.keypoint_names[62]) == (
        "right_thumb4", "right_wrist", "left_thumb4", "left_wrist",
    )
    assert all(0 <= a < 70 and 0 <= b < 70 for a, b in MHR_70.links)


def test_body_poses_count_people() -> None:
    assert body_rows(2, 2).num_people == 2
    assert body_rows(0, 0).num_people == 0


def test_body_poses_reject_misaligned_boxes() -> None:
    with pytest.raises(ValueError, match="rows disagree"):
        body_rows(2, 1)
