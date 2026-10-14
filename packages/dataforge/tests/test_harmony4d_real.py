"""Golden checks on the real release (``test/01_hugging.zip``, 1.7 GB): our reading of the calibration and of the SMPL fits
reproduces what the release ships.

* Projection: the shipped SMPL-45 joints through our exo and Aria lenses land on the shipped ``poses2d``, which the release made
  the same way. Exo cameras differ by up to ~2 px because the release poses each frame with the nearest of a camera's few COLMAP
  registrations and we keep the earliest; the Aria cameras by ~1 px more because the release turns its images with ``x' = W - y``
  where projectaria-tools (and the JPEG pixels) use ``W - 1 - y``.
* Body: the official neutral SMPL evaluated on the shipped parameters gives the shipped vertices.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import raw_asset
from simplecv.ops.smplx.smplx_torch import SmplxLayerTorch
from simplecv.sensors.camera.fisheye62 import project_fisheye62

from dataforge import paths
from dataforge.datasets.egohumans_layers import exo_camera, exo_scale
from dataforge.datasets.egohumans_source import ARIA_MODES, Sequence, ZipSequenceFiles, load_pickle, read_sequence, sequences_in

pytestmark = pytest.mark.golden

ROOT: Path = Path(os.environ.get("DATAFORGE_HARMONY4D_ROOT", paths.raw_root() / "harmony4d"))
"""The release root; a full run deletes each zip once converted, so the golden copy may live elsewhere."""
FRAMES_CHECKED: tuple[int, ...] = (0, 150, 300)


@pytest.fixture(scope="module")
def release() -> tuple[ZipSequenceFiles, Sequence]:
    archive = zipfile.ZipFile(raw_asset("Harmony4D test/01_hugging.zip (dataforge-download harmony4d --sequences test/01_hugging)", ROOT / "test/01_hugging.zip"))
    files = sequences_in(archive)["01_hugging/002_hugging"]
    return files, read_sequence(files)


def compare(ours: np.ndarray, shipped: np.ndarray, width: int, height: int) -> tuple[float, int, int]:
    """(largest pixel offset where we project, shipped joints well inside the image, how many of those we do not project)."""
    visible = np.isfinite(ours).all(axis=1)
    inside = (shipped[:, 0] >= 5) & (shipped[:, 0] < width - 5) & (shipped[:, 1] >= 5) & (shipped[:, 1] < height - 5)
    offset = float(np.abs(ours[visible] - shipped[visible, :2]).max()) if visible.any() else 0.0
    return offset, int(inside.sum()), int((inside & ~visible).sum())


def check(results: list[tuple[float, int, int]], max_px: float, max_missing: float) -> None:
    """Every projected joint within ``max_px``, and at most ``max_missing`` of the shipped joints inside the image unprojected."""
    inside = sum(result[1] for result in results)
    missing = sum(result[2] for result in results)
    assert inside > 0 and missing <= max_missing * inside, f"{missing} of {inside} shipped joints inside the image are not projected"
    assert max(result[0] for result in results) < max_px


def test_exo_lenses_reproduce_the_shipped_2d(release: tuple[ZipSequenceFiles, Sequence]) -> None:
    files, sequence = release
    results: list[tuple[float, int, int]] = []
    for row in FRAMES_CHECKED:
        frame = sequence.frames[row]
        for camera in sequence.exo:
            shipped = load_pickle(files.read(f"processed_data/poses2d/{camera.name}/rgb/{frame:05d}.npy"), camera.name)
            assert isinstance(shipped, dict)
            stored = exo_camera(camera)  # the lens base logs; the shipped 2D is at the source size
            cam_T_world = stored.extrinsics.cam_T_world
            for person in sequence.people:
                joints = person.joints[row].astype(np.float64) @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
                results.append(compare(project_fisheye62(joints, stored) / exo_scale(camera), np.asarray(shipped[person.name]), camera.width, camera.height))
    assert len(sequence.exo) == 22
    assert np.median([result[0] for result in results]) < 0.5
    check(results, 2.5, 0.0)


def test_aria_lenses_reproduce_the_shipped_2d(release: tuple[ZipSequenceFiles, Sequence]) -> None:
    files, sequence = release
    results: list[tuple[float, int, int]] = []
    for row in FRAMES_CHECKED:
        frame = sequence.frames[row]
        for glasses in sequence.glasses:
            for index, mode in enumerate(ARIA_MODES):
                shipped = load_pickle(files.read(f"processed_data/poses2d/{glasses.name}/{mode}/{frame:05d}.npy"), mode)
                assert isinstance(shipped, dict)
                for person, joints in ((person, person.joints[row]) for person in sequence.people if person.name in shipped):
                    upright = glasses.cameras[index].calibration(row)
                    pixels = glasses.cameras[index].project(row, joints.astype(np.float64))
                    results.append(compare(pixels, np.asarray(shipped[person.name]), upright.width, upright.height))
    assert [glasses.name for glasses in sequence.glasses] == ["aria01", "aria02"]
    # 12 of 453 are not projected, all just past projectaria-tools' Gen1 cone or valid radius (rgb 57-64 deg off-axis, slam
    # 80-88 deg), which the release's 2D does not apply.
    check(results, 2.0, 0.05)


def test_official_neutral_smpl_reproduces_the_shipped_vertices(release: tuple[ZipSequenceFiles, Sequence]) -> None:
    _, sequence = release
    model_root: Path = raw_asset("official neutral SMPL", ROOT / "body_models/smpl/SMPL_NEUTRAL.pkl").parents[1]
    for person in sequence.people:
        rows = np.flatnonzero(person.present)[:40]
        model = SmplxLayerTorch(betas=person.betas[rows[0]], model_type="smpl", gender="neutral", model_root_dir=model_root)
        poses = np.concatenate([person.global_orient[rows], person.body_pose[rows]], axis=1)
        vertices = model.forward_batched(poses, person.transl[rows]).vertices
        assert np.abs(vertices - person.vertices[rows]).max() < 1e-5  # metres: 0.01 mm
