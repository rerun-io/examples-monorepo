"""Bystanders on FullCircle's level faces: face masks mapped back to a source camera, SAM 3 instances a YOLOX box confirms."""

from __future__ import annotations

from pathlib import Path

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


def test_faces_render_the_source_pixels_each_face_looks_at() -> None:
    from posekit.models.fullcircle import sample_faces
    from posekit.ops.sphere import EquirectCamera, PinholeCamera, pixel_grid

    # Panorama dark left of straight ahead (x < 0), bright right of it.
    width, height = 96, 48
    source: EquirectCamera = EquirectCamera(width, height)
    rays, _ = source.unproject(pixel_grid(height, width, torch.device("cpu")))
    image: torch.Tensor = torch.where(rays[:, 0] > 0, 200.0, 0.0).float().reshape(1, height, width).expand(3, -1, -1)
    face_rays, _ = PinholeCamera(32).unproject(pixel_grid(32, 32, torch.device("cpu")))
    behind: torch.Tensor = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    faces: torch.Tensor = sample_faces(
        [image], face_rays.float(), torch.stack([torch.eye(3), behind]), 32, [source], torch.eye(3)[None], [(height, width)]
    )
    assert faces.shape == (2, 32, 32, 3) and faces.dtype == torch.uint8
    # Ahead: left half dark, right half bright; behind, the face's right is the rig's left.
    assert int(faces[0, :, :12].max()) == 0 and int(faces[0, :, 20:].min()) == 200
    assert int(faces[1, :, :12].min()) == 200 and int(faces[1, :, 20:].max()) == 0


def test_mask_radius_is_the_dilation_in_mask_cells() -> None:
    from posekit.models.fullcircle import SphereSource
    from posekit.ops.sphere import EquirectCamera

    source: SphereSource = SphereSource(EquirectCamera(1920, 960), np.eye(3), (960, 1920), (240, 480))
    assert [source.mask_radius(px) for px in (0.0, 8.0, 12.0, 13.0)] == [0, 2, 3, 4]


def test_people_refuse_the_sam3_operator_detector() -> None:
    import pytest

    from posekit.apis.lens_masks import LensMasksConfig
    from posekit.models.fullcircle import FullCircleConfig
    from posekit.models.fullcircle_people import FacePeopleConfig

    with pytest.raises(ValueError, match="YOLOX"):
        LensMasksConfig(output_dir=Path("layers"), masker=FullCircleConfig(detector="sam3"))
    LensMasksConfig(output_dir=Path("layers"), masker=FullCircleConfig(detector="sam3"), people=FacePeopleConfig(enabled=False))


def test_the_lens_mask_step_tracks_people_on_its_own_framesets() -> None:
    """One decode serves FullCircle and the people tracker, so the tracker has no stride of its own."""
    import pytest

    from posekit.apis.lens_masks import LensMasksConfig
    from posekit.apis.people_tracks import TrackOptions

    with pytest.raises(ValueError, match="track.stride"):
        LensMasksConfig(output_dir=Path("layers"), track=TrackOptions(stride=5))
