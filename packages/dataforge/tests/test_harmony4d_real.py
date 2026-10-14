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

from dataforge import paths
from dataforge.datasets.egohumans_source import ARIA_MODES, Sequence, ZipSequenceFiles, load_pickle, read_sequence

pytestmark = pytest.mark.golden

ROOT: Path = Path(os.environ.get("DATAFORGE_HARMONY4D_ROOT", paths.raw_root() / "harmony4d"))
"""The release root; a full run deletes each zip once converted, so the golden copy may live elsewhere."""
FRAMES_CHECKED: tuple[int, ...] = (0, 150, 300)


@pytest.fixture(scope="module")
def release() -> tuple[ZipSequenceFiles, Sequence]:
    archive = zipfile.ZipFile(raw_asset("Harmony4D test/01_hugging.zip (dataforge-download harmony4d --sequences test/01_hugging)", ROOT / "test/01_hugging.zip"))
    files = ZipSequenceFiles(archive, "01_hugging/002_hugging")
    return files, read_sequence(files)


def max_offset(ours: np.ndarray, shipped: np.ndarray) -> float:
    visible = np.isfinite(ours).all(axis=1)
    return float(np.abs(ours[visible] - shipped[visible, :2]).max()) if visible.any() else 0.0


def test_exo_lenses_reproduce_the_shipped_2d(release: tuple[ZipSequenceFiles, Sequence]) -> None:
    files, sequence = release
    offsets: list[float] = []
    for row in FRAMES_CHECKED:
        frame = sequence.frames[row]
        for camera in sequence.exo:
            shipped = load_pickle(files.read(f"processed_data/poses2d/{camera.name}/rgb/{frame:05d}.npy"), camera.name)
            assert isinstance(shipped, dict)
            for person in sequence.people:
                offsets.append(max_offset(camera.project(person.joints[row].astype(np.float64)), np.asarray(shipped[person.name])))
    assert len(sequence.exo) == 22
    assert np.median(offsets) < 0.5 and max(offsets) < 2.5


def test_aria_lenses_reproduce_the_shipped_2d(release: tuple[ZipSequenceFiles, Sequence]) -> None:
    files, sequence = release
    offsets: list[float] = []
    for row in FRAMES_CHECKED:
        frame = sequence.frames[row]
        for glasses in sequence.glasses:
            for index, mode in enumerate(ARIA_MODES):
                shipped = load_pickle(files.read(f"processed_data/poses2d/{glasses.name}/{mode}/{frame:05d}.npy"), mode)
                assert isinstance(shipped, dict)
                for person, joints in ((person, person.joints[row]) for person in sequence.people if person.name in shipped):
                    offsets.append(max_offset(glasses.cameras[index].project(row, joints.astype(np.float64)), np.asarray(shipped[person.name])))
    assert [glasses.name for glasses in sequence.glasses] == ["aria01", "aria02"]
    assert offsets and max(offsets) < 2.0


def test_official_neutral_smpl_reproduces_the_shipped_vertices(release: tuple[ZipSequenceFiles, Sequence]) -> None:
    _, sequence = release
    model_root: Path = raw_asset("official neutral SMPL", ROOT / "body_models/smpl/SMPL_NEUTRAL.pkl").parents[1]
    for person in sequence.people:
        rows = np.flatnonzero(person.present)[:40]
        model = SmplxLayerTorch(betas=person.betas[rows[0]], model_type="smpl", gender="neutral", model_root_dir=model_root)
        poses = np.concatenate([person.global_orient[rows], person.body_pose[rows]], axis=1)
        vertices = model.forward_batched(poses, person.transl[rows]).vertices
        assert np.abs(vertices - person.vertices[rows]).max() < 1e-5  # metres: 0.01 mm
