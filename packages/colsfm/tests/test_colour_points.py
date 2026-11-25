"""Point colours from the run's frames: the mean bilinear sample over each point's observations."""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
import torch
from conftest import build_reconstruction
from mapping_helpers import FixedFrames, SyntheticRig

from colsfm.frames import colour_points


def _reference_colours(reconstruction: pycolmap.Reconstruction, frames: FixedFrames) -> dict[int, np.ndarray]:
    """The rule written out point by point: round(mean over observations of the bilinear sample at xy - 0.5)."""
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for image_id in sorted(reconstruction.reg_image_ids()):
        image = reconstruction.image(image_id)
        rgb: np.ndarray = frames.frames_by_name[image.name].numpy().astype(np.float64)
        height, width = rgb.shape[1:]
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            x, y = point2D.xy - 0.5
            if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
                continue
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
            fx, fy = x - x0, y - y0
            value = rgb[:, y0, x0] * (1 - fx) * (1 - fy) + rgb[:, y0, x1] * fx * (1 - fy) + rgb[:, y1, x0] * (1 - fx) * fy + rgb[:, y1, x1] * fx * fy
            sums[point2D.point3D_id] = sums.get(point2D.point3D_id, np.zeros(3)) + value
            counts[point2D.point3D_id] = counts.get(point2D.point3D_id, 0) + 1
    return {point_id: np.clip(np.rint(sums[point_id] / counts[point_id]), 0, 255).astype(np.uint8) for point_id in sums}


def test_every_point_gets_the_rounded_mean_of_its_observation_samples(synthetic_rig: SyntheticRig) -> None:
    """CPU only (the unit lane): the synthetic rig's poses, random RGB frames, the rule written out per point as reference."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta, error_px=0.3)
    sizes: dict[str, tuple[int, int]] = {
        image.name: (reconstruction.camera(image.camera_id).width, reconstruction.camera(image.camera_id).height)
        for image in reconstruction.images.values()
    }
    frames = FixedFrames(sizes, random=True)
    expected: dict[int, np.ndarray] = _reference_colours(reconstruction, frames)
    coloured: int = colour_points(reconstruction, frames)
    assert expected
    assert coloured == sum(int(colour.any()) for colour in expected.values())
    mismatched: list[int] = [
        point_id
        for point_id, colour in expected.items()
        if np.abs(reconstruction.points3D[point_id].color.astype(int) - colour.astype(int)).max() > 1
    ]
    assert not mismatched


def _colour_points_before_index_add(reconstruction: pycolmap.Reconstruction, frames: FixedFrames) -> int:
    """`colour_points` as it was before the device-side accumulation (every Point2D, `.cpu()` and `np.add.at` per image)."""
    point_ids: np.ndarray = np.asarray(sorted(reconstruction.points3D), dtype=np.int64)
    sums: np.ndarray = np.zeros((len(point_ids), 3), dtype=np.float64)
    counts: np.ndarray = np.zeros(len(point_ids), dtype=np.int64)
    image_ids: list[int] = sorted(int(image_id) for image_id in reconstruction.reg_image_ids())
    names: list[str] = [reconstruction.image(image_id).name for image_id in image_ids]
    for image_id, (_, rgb) in zip(image_ids, frames.frames(names), strict=True):
        observed = [(point2D.xy, point2D.point3D_id) for point2D in reconstruction.image(image_id).points2D if point2D.has_point3D()]
        if not observed:
            continue
        xy: np.ndarray = np.asarray([pixel for pixel, _ in observed], dtype=np.float64)
        rows: np.ndarray = np.searchsorted(point_ids, np.asarray([point_id for _, point_id in observed], dtype=np.int64))
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        centre: np.ndarray = xy - 0.5
        inside: np.ndarray = (centre[:, 0] >= 0) & (centre[:, 0] <= width - 1) & (centre[:, 1] >= 0) & (centre[:, 1] <= height - 1)
        if not inside.any():
            continue
        grid = torch.from_numpy((2.0 * xy[inside] / np.array([width, height]) - 1.0).astype(np.float32)).to(rgb.device)[None, None]
        sampled: np.ndarray = torch.nn.functional.grid_sample(rgb[None].float(), grid, mode="bilinear", align_corners=False)[0, :, 0].T.cpu().numpy()
        np.add.at(sums, rows[inside], sampled.astype(np.float64))
        np.add.at(counts, rows[inside], 1)
    coloured: int = 0
    seen: np.ndarray = np.flatnonzero(counts)
    colours: np.ndarray = np.clip(np.rint(sums[seen] / counts[seen, None]), 0, 255).astype(np.uint8)
    for point_id, colour in zip(point_ids[seen], colours, strict=True):
        reconstruction.points3D[int(point_id)].color = colour
        coloured += int(colour.any())
    return coloured


def _scene_with_awkward_observations(synthetic_rig: SyntheticRig) -> pycolmap.Reconstruction:
    """The synthetic rig plus a point seen twice in one image, coloured from frames 3/4 of the camera's size, so a
    quarter of the observations fall outside the frame."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta, error_px=0.3)
    image_id: int = sorted(reconstruction.reg_image_ids())[0]
    free: list[int] = [index for index, point2D in enumerate(reconstruction.image(image_id).points2D) if not point2D.has_point3D()][:2]
    if len(free) == 2:
        reconstruction.add_point3D(np.zeros(3), pycolmap.Track([pycolmap.TrackElement(image_id, index) for index in free]))
    return reconstruction


def _shrunk_sizes(reconstruction: pycolmap.Reconstruction) -> dict[str, tuple[int, int]]:
    return {
        image.name: (3 * reconstruction.camera(image.camera_id).width // 4, 3 * reconstruction.camera(image.camera_id).height // 4)
        for image in reconstruction.images.values()
    }


def test_colours_equal_the_per_image_host_accumulation_bit_for_bit_on_cpu(synthetic_rig: SyntheticRig) -> None:
    """CPU frames: the device-side `index_add_` gives exactly the colours the host `np.add.at` gave, a point seen twice
    in one image and observations outside the frame included (float64 sums added in the same order)."""
    expected_model: pycolmap.Reconstruction = _scene_with_awkward_observations(synthetic_rig)
    actual_model: pycolmap.Reconstruction = _scene_with_awkward_observations(synthetic_rig)
    frames = FixedFrames(_shrunk_sizes(expected_model), random=True)
    expected: int = _colour_points_before_index_add(expected_model, frames)
    assert colour_points(actual_model, frames) == expected
    assert expected > 0
    outside: int = sum(
        1
        for image in expected_model.images.values()
        for point2D in image.get_observation_points2D()
        if not (0 <= point2D.xy[0] - 0.5 <= frames.frames_by_name[image.name].shape[2] - 1)
    )
    assert outside > 0, "some observations fall outside the shrunk frames"
    for point_id, point in expected_model.points3D.items():
        assert np.array_equal(actual_model.points3D[point_id].color, point.color), point_id


@pytest.mark.integration
def test_colours_on_cuda_frames_match_the_cpu_ones_within_one_level(synthetic_rig: SyntheticRig) -> None:
    """CUDA frames: the sums accumulate on the device (atomic adds, any order), so a colour may round 1 level apart."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    cpu_model: pycolmap.Reconstruction = _scene_with_awkward_observations(synthetic_rig)
    cuda_model: pycolmap.Reconstruction = _scene_with_awkward_observations(synthetic_rig)
    frames = FixedFrames(_shrunk_sizes(cpu_model), random=True)
    expected: int = colour_points(cpu_model, frames)
    frames.frames_by_name = {name: rgb.cuda() for name, rgb in frames.frames_by_name.items()}
    assert abs(colour_points(cuda_model, frames) - expected) <= 1
    for point_id, point in cpu_model.points3D.items():
        assert np.abs(cuda_model.points3D[point_id].color.astype(int) - point.color.astype(int)).max() <= 1, point_id
