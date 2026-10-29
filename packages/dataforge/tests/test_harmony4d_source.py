"""The Harmony4D reader: world mapping, rigid poses, lens projections, people and the pickle door, on a synthetic sequence."""

from __future__ import annotations

import io
import os
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from harmony4d_synthetic import (
    ARIA_FOCAL,
    ARIA_RGB_UPRIGHT,
    ARIA_SLAM_OFFSETS,
    COLMAP_FROM_WORLD,
    EXO_LENS,
    EXO_POSES,
    FRAMES,
    npy_pickle,
    person_pose,
    write_sequence,
    zip_sequence,
)
from simplecv.sensors.camera.fisheye62 import project_fisheye62

from dataforge.datasets.harmony4d_layers import aria_camera, ego_rig, encode_job, exo_camera, exo_scale, person_index
from dataforge.datasets.harmony4d_source import PersonTrack, Sequence, ZipSequenceFiles, load_pickle, read_aria_calibration, read_sequence
from dataforge.video_encoding import halvings


@pytest.fixture
def files(tmp_path: Path) -> ZipSequenceFiles:
    return zip_sequence(write_sequence(tmp_path / "01_hugging" / "002_hugging"))


@pytest.fixture
def sequence(files: ZipSequenceFiles) -> Sequence:
    return read_sequence(files)


def test_exo_poses_are_rigid_and_metric_despite_colmap_scale(sequence: Sequence) -> None:
    assert [camera.name for camera in sequence.exo] == ["cam01", "cam02"]
    for camera in sequence.exo:
        np.testing.assert_allclose(camera.world_T_cam, EXO_POSES[camera.name], atol=1e-9)
        np.testing.assert_allclose(camera.lens, EXO_LENS)
    assert sequence.world_source == "colmap_from_aria_transforms.pkl"


def test_exo_lens_at_the_stored_size_puts_the_optical_axis_at_the_principal_point(sequence: Sequence) -> None:
    camera = sequence.exo[0]
    stored = exo_camera(camera)
    assert (stored.intrinsics.width, stored.intrinsics.height) == (1920, 1080)
    cam_T_world = stored.extrinsics.cam_T_world
    ahead = camera.world_T_cam[:3, 3] + 2.0 * camera.world_T_cam[:3, 2]
    behind = camera.world_T_cam[:3, 3] - 2.0 * camera.world_T_cam[:3, 2]
    pixels = project_fisheye62(np.stack([ahead, behind]) @ cam_T_world[:3, :3].T + cam_T_world[:3, 3], stored)
    np.testing.assert_allclose(pixels[0], np.array(EXO_LENS[2:4]) * exo_scale(camera), atol=1e-6)
    assert np.isnan(pixels[1]).all()


def test_aria_frames_are_turned_upright(sequence: Sequence) -> None:
    (glasses,) = sequence.glasses
    rgb = glasses.cameras[0]
    width, height = ARIA_RGB_UPRIGHT
    # The stored image is upright, so the device-orientation image is height x width.
    assert (rgb.width, rgb.height) == (height, width)
    # A point to the device camera's right (+x, world +x) lands below the centre of the upright image:
    # a quarter turn clockwise maps device (u, v) to (H - 1 - v, u).
    cx = (height - 1) / 2
    cy = (width - 1) / 2
    point = np.array([[0.2, 1.0, 0.0]])  # world: 1 m ahead (+y), 0.2 m to the right (+x)
    expected_u = cx + (ARIA_FOCAL + 0.001) * np.arctan(0.2)
    np.testing.assert_allclose(rgb.project(0, point)[0], [height - 1 - cy, expected_u], atol=1e-6)


def test_aria_rig_is_the_rgb_camera_and_slams_sit_beside_it(sequence: Sequence) -> None:
    (glasses,) = sequence.glasses
    assert glasses.world_T_rig.shape == (len(FRAMES), 4, 4)
    for mode, offset in ARIA_SLAM_OFFSETS.items():
        np.testing.assert_allclose(glasses.rig_T_cam(mode)[:3, 3], glasses.world_T_rig[0, :3, :3].T @ offset, atol=1e-9)
    np.testing.assert_allclose(glasses.cameras[0].intrinsics[:, 0], ARIA_FOCAL + 0.001 * np.array(FRAMES))


def test_people_cover_every_frame_with_nan_where_unlabelled(sequence: Sequence) -> None:
    assert [person.name for person in sequence.people] == ["aria01", "aria02"]
    aria01, aria02 = sequence.people
    assert aria01.present.all()
    assert aria02.present.tolist() == [True, True, False]
    assert np.isnan(aria02.coco17[-1]).all() and (aria02.coco17_confidence[-1] == 0.0).all()
    np.testing.assert_allclose(aria01.coco17_confidence, 0.9)
    np.testing.assert_allclose(aria01.betas, 0.1)
    assert aria01.fit_loss.tolist() == [0.5] * len(FRAMES)
    assert [person_index(person) for person in sequence.people] == [0, 1]


def test_reader_allocates_each_person_track_once(files: ZipSequenceFiles, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole-sequence vertex buffers must not be allocated again for each label."""
    allocated: list[str] = []
    original = PersonTrack.empty

    def empty(name: str, frames: int) -> PersonTrack:
        allocated.append(name)
        return original(name, frames)

    monkeypatch.setattr(PersonTrack, "empty", empty)
    sequence = read_sequence(files)
    assert allocated == [person.name for person in sequence.people]


@pytest.mark.parametrize("flat", [False, True])
def test_boxes_are_read_per_camera_and_person_in_every_layout(tmp_path: Path, flat: bool) -> None:
    sequence = read_sequence(zip_sequence(write_sequence(tmp_path / "s" / "q", flat_boxes=flat)))
    (boxes,) = sequence.boxes
    assert boxes.camera == "cam01"
    np.testing.assert_array_equal(boxes.boxes["aria01"][0], [10, 20, 110, 220])
    assert np.isnan(boxes.boxes["aria02"][-1]).all()


def test_exo_only_take_reaches_the_world_through_scale(tmp_path: Path) -> None:
    sequence = read_sequence(zip_sequence(write_sequence(tmp_path / "s" / "q", glasses=False)))
    assert sequence.glasses == [] and sequence.glasses_left_out == [] and sequence.world_source == "scale.npy"
    for camera in sequence.exo:
        np.testing.assert_allclose(camera.world_T_cam, EXO_POSES[camera.name], atol=1e-9)


def test_scale_decides_the_world_and_leaves_the_glasses_out(sequence: Sequence, tmp_path: Path) -> None:
    # As 29 release takes ship (train/11_karate3, 12_mma, 13_mma2, 14_mma3): glasses, their transforms and scale.npy. The world
    # is scale.npy's and, as in upstream's loader, the glasses are not placed: they are named as left out instead.
    sequence_dir = write_sequence(tmp_path / "s" / "q")
    np.save(sequence_dir / "colmap/workplace/scale.npy", np.linalg.inv(COLMAP_FROM_WORLD))
    scaled = read_sequence(zip_sequence(sequence_dir))
    assert scaled.world_source == "scale.npy" and scaled.glasses == [] and scaled.glasses_left_out == ["aria01"]
    assert [person.name for person in scaled.people] == [person.name for person in sequence.people]
    np.testing.assert_allclose(scaled.exo[0].world_T_cam, EXO_POSES["cam01"], atol=1e-9)
    assert all(not boxes.camera.startswith("aria") for boxes in scaled.boxes)



def test_a_fit_missing_a_field_names_its_file_and_person(tmp_path: Path) -> None:
    sequence_dir = write_sequence(tmp_path / "s" / "q")
    fit: dict[str, object] = person_pose("aria02", 2)[1]
    del fit["betas"]
    npy_pickle(sequence_dir / "processed_data" / "smpl" / "00002.npy", {"aria01": person_pose("aria01", 2)[1], "aria02": fit})
    with pytest.raises(ValueError, match="smpl/00002.npy: aria02"):
        read_sequence(zip_sequence(sequence_dir))

def test_pickles_naming_other_globals_are_refused() -> None:
    class Payload:
        def __reduce__(self) -> tuple[Callable[[], str], tuple[()]]:
            return (os.getcwd, ())  # anything outside numpy: refused before it runs

    with pytest.raises(ValueError, match="not a numpy type"):
        load_pickle(pickle.dumps({"aria01": Payload()}), "evil.npy")
    buffer = io.BytesIO()
    np.save(buffer, np.array({"aria01": np.arange(3)}, dtype=object), allow_pickle=True)
    loaded = load_pickle(buffer.getvalue(), "labels.npy")
    assert isinstance(loaded, dict)
    np.testing.assert_array_equal(loaded["aria01"], np.arange(3))
    plain = io.BytesIO()
    np.save(plain, np.eye(4))
    np.testing.assert_array_equal(load_pickle(plain.getvalue(), "scale.npy"), np.eye(4))


def test_calibration_with_two_people_is_refused() -> None:
    with pytest.raises(ValueError, match="expected 7"):
        read_aria_calibration("header\n" + "aria\n1\n2\n" * 4, "two.txt")


def test_rig_slots_and_stored_sizes(sequence: Sequence) -> None:
    assert [camera.number for camera in sequence.exo] == [1, 2]
    assert [ego_rig(glasses) for glasses in sequence.glasses] == [91]
    assert halvings(3840, 2160) == 1 and halvings(1408, 1408) == 0 and halvings(480, 640) == 0


def test_encode_refuses_frames_that_do_not_match_the_logged_lens(sequence: Sequence, files: ZipSequenceFiles, tmp_path: Path) -> None:
    pytest.importorskip("turbojpeg", reason="PyTurboJPEG is a dataforge-env dependency")
    job = encode_job(files, "exo/cam01/images", sequence.frames, tmp_path / "cam01.mp4", (1, 2), aria_camera(sequence.glasses[0], 0))
    with pytest.raises(ValueError, match="the logged lens is"):
        job()
