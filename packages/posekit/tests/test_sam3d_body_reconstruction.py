"""Row selection and the empty case of ``Sam3dBodyReconstruction`` (CPU-only, synthetic rows)."""

from dataclasses import fields

import pytest
import torch
from torch import Tensor

from posekit.models.sam3d_body import MHR_PARAMETER_WIDTHS, ROOT_INDICES, Sam3dBodyReconstruction
from posekit.predictions import Keypoints3d
from posekit.skeletons import MHR_70

JOINTS: int = 127
VERTICES: int = 18


def people(count: int, *, hands: bool) -> Sam3dBodyReconstruction:
    """``count`` rows whose every value encodes its row, so a wrong selection shows."""

    def rows(*shape: int) -> Tensor:
        return torch.arange(count, dtype=torch.float32).reshape(count, *([1] * len(shape))).expand(count, *shape).clone()

    return Sam3dBodyReconstruction(
        keypoints=Keypoints3d(
            rows(70, 2), rows(70), rows(70), torch.arange(count, dtype=torch.int64), MHR_70, ROOT_INDICES, rows(70, 3)
        ),
        boxes=rows(4),
        vertices=rows(VERTICES, 3),
        faces=torch.arange(12, dtype=torch.int64).reshape(4, 3),
        translation=rows(3),
        focal_length=rows(),
        global_rotation=rows(3),
        **{name: rows(width) for name, width in MHR_PARAMETER_WIDTHS.items()},
        joint_coordinates=rows(JOINTS, 3),
        joint_rotations=rows(JOINTS, 3, 3),
        left_hand_boxes=rows(4) if hands else None,
        right_hand_boxes=rows(4) if hands else None,
    )


@pytest.mark.parametrize("hands", [True, False])
def test_take_selects_every_row_field(hands: bool) -> None:
    source: Sam3dBodyReconstruction = people(3, hands=hands)
    selected: Sam3dBodyReconstruction = source.take(torch.tensor([2, 0]))
    assert selected.num_people == 2
    assert selected.faces is source.faces
    assert (selected.keypoints.skeleton, selected.keypoints.root_indices) == (MHR_70, ROOT_INDICES)
    for record, original in ((selected, source), (selected.keypoints, source.keypoints)):
        for field in fields(record):
            value = getattr(record, field.name)
            if field.name == "faces" or not isinstance(value, Tensor):
                continue
            assert torch.equal(value, getattr(original, field.name)[[2, 0]]), field.name
    assert (selected.left_hand_boxes is None) is not hands


@pytest.mark.parametrize("hands", [True, False])
def test_empty_has_the_shapes_of_a_row(hands: bool) -> None:
    faces: Tensor = torch.arange(12, dtype=torch.int64).reshape(4, 3)
    empty: Sam3dBodyReconstruction = Sam3dBodyReconstruction.empty(faces, num_vertices=VERTICES, num_mhr_joints=JOINTS, hands=hands, device="cpu")
    row: Sam3dBodyReconstruction = people(1, hands=hands)
    assert empty.num_people == 0
    assert empty.faces is faces
    for record, reference in ((empty, row), (empty.keypoints, row.keypoints)):
        for field in fields(record):
            value = getattr(record, field.name)
            if field.name == "faces" or not isinstance(value, Tensor):
                assert (value is None) == (getattr(reference, field.name) is None), field.name
                continue
            assert value.shape[1:] == getattr(reference, field.name).shape[1:], field.name
            assert value.dtype == getattr(reference, field.name).dtype, field.name
