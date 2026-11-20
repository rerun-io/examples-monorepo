"""Point colours from the run's frames: the mean bilinear sample over each point's observations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pycolmap
import torch
from conftest import build_reconstruction
from mapping_helpers import SyntheticRig
from torch import Tensor

from colsfm.frames import colour_points


class _RandomFrames:
    """A frame source of fixed random RGB images, one per name, at each image's camera size."""

    def __init__(self, sizes: dict[str, tuple[int, int]]) -> None:
        self.frames_by_name: dict[str, Tensor] = {
            name: torch.from_numpy(np.random.default_rng(index).integers(0, 256, (3, height, width), dtype=np.uint8))
            for index, (name, (width, height)) in enumerate(sorted(sizes.items()))
        }

    def frames(self, names: Sequence[str]) -> Iterator[tuple[str, Tensor]]:
        for name in names:
            yield name, self.frames_by_name[name]

    def close(self) -> None:
        """Nothing to release."""


def _reference_colours(reconstruction: pycolmap.Reconstruction, frames: _RandomFrames) -> dict[int, np.ndarray]:
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
            value = (rgb[:, y0, x0] * (1 - fx) * (1 - fy) + rgb[:, y0, x1] * fx * (1 - fy) + rgb[:, y1, x0] * (1 - fx) * fy + rgb[:, y1, x1] * fx * fy)
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
    frames = _RandomFrames(sizes)
    expected: dict[int, np.ndarray] = _reference_colours(reconstruction, frames)
    coloured: int = colour_points(reconstruction, frames)
    assert expected
    assert coloured == sum(int(colour.any()) for colour in expected.values())
    mismatched: list[int] = [point_id for point_id, colour in expected.items() if np.abs(reconstruction.points3D[point_id].color.astype(int) - colour.astype(int)).max() > 1]
    assert not mismatched
