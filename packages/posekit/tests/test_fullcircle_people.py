"""Bystanders on FullCircle's level faces: face masks mapped back to a source camera, SAM 3 instances a YOLOX box confirms."""

from __future__ import annotations

import numpy as np
import torch


def test_a_face_mask_maps_back_to_the_rays_inside_that_face() -> None:
    from posekit.models.fullcircle_people import faces_to_source_mask
    from posekit.ops.sphere import EquirectCamera, PinholeCamera, pixel_grid

    source: EquirectCamera = EquirectCamera(96, 48)
    rays, _ = source.unproject(pixel_grid(48, 96, torch.device("cpu")))
    face: PinholeCamera = PinholeCamera(32)
    # The +Z face with its left half (x < 0) masked: only rays ahead (z > |x|, |y|) and to the left come back.
    masks: torch.Tensor = torch.zeros(1, 32, 32, dtype=torch.bool)
    masks[0, :, :16] = True
    hit: torch.Tensor = faces_to_source_mask(masks, torch.eye(3)[None], face, rays.float())
    x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
    inside: torch.Tensor = (z > x.abs()) & (z > y.abs())
    assert bool(hit[inside & (x < -0.05)].all())
    assert not bool(hit[~inside].any())
    assert not bool(hit[x > 0.05].any())


def test_a_face_rotated_to_minus_z_masks_the_rays_behind() -> None:
    from posekit.models.fullcircle_people import faces_to_source_mask
    from posekit.ops.sphere import EquirectCamera, PinholeCamera, pixel_grid

    source: EquirectCamera = EquirectCamera(96, 48)
    rays, _ = source.unproject(pixel_grid(48, 96, torch.device("cpu")))
    rig_R_face: torch.Tensor = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    hit: torch.Tensor = faces_to_source_mask(torch.ones(1, 32, 32, dtype=torch.bool), rig_R_face[None], PinholeCamera(32), rays.float())
    z: torch.Tensor = rays[:, 2]
    assert bool(hit[z < -0.9].all()) and not bool(hit[z > 0].any())


def test_only_instances_a_bystander_box_confirms_are_kept() -> None:
    from posekit.models.fullcircle_people import match_instances

    candidates: np.ndarray = np.array([[100.0, 100.0, 120.0, 150.0]], dtype=np.float32)
    instances: np.ndarray = np.array([[101.0, 102.0, 121.0, 149.0], [300.0, 300.0, 340.0, 400.0], [100.0, 100.0, 200.0, 300.0]], dtype=np.float32)
    keep: np.ndarray = match_instances(candidates, instances, min_iou=0.3)
    assert keep.tolist() == [True, False, False]
    assert match_instances(np.zeros((0, 4), dtype=np.float32), instances, min_iou=0.3).tolist() == [False, False, False]
