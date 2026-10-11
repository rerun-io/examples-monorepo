"""MammaSyn shard readers against hand-built WebDataset tars laid out as the release ships them."""

import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

from dataforge.datasets.mamma_source import MammaCamera
from dataforge.datasets.mamma_syn_source import RenderLabels, RenderView, SynSample, assemble_render, index_shards, read_member, read_render_view

RENDER_PREFIX: str = "/tmp/bedlam_lab_20251031_191434/interhand_2-6_NC_200_00/png/be_NB-NZj2zRkeQ_seq_000002"
BEDLAM_PREFIX: str = "/ps/project/bedlam/public/b0_all/20221010_3-10_500_batch01hand_zoom_suburb_d/png"
KINDS: tuple[str, ...] = ("png", "mask.png", "data.pyd")


def write_shard(path: Path, members: list[tuple[str, bytes]]) -> Path:
    """A plain (uncompressed) tar holding ``members`` in the given order, as the release's shards are."""
    with tarfile.open(path, "w") as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def sample_members(key: str) -> list[tuple[str, bytes]]:
    return [(f"{key}.{kind}", f"{key} {kind}".encode()) for kind in KINDS]


def test_render_shards_index_one_sample_per_camera_and_frame_in_frame_order(tmp_path: Path) -> None:
    # A camera straddles the shard boundary, and frame 2 is missing in every camera, as in the sampled Hands sequence.
    first: Path = write_shard(
        tmp_path / "000000.tar",
        [member for frame in (0, 1, 3) for member in sample_members(f"{RENDER_PREFIX}/IOI_18/{frame:04d}")]
        + [member for frame in (0,) for member in sample_members(f"{RENDER_PREFIX}/IOI_01/{frame:04d}")],
    )
    second: Path = write_shard(tmp_path / "000001.tar", [member for frame in (3, 1) for member in sample_members(f"{RENDER_PREFIX}/IOI_01/{frame:04d}")])
    samples: list[SynSample] = index_shards([first, second])
    assert [(sample.camera, sample.frame) for sample in samples] == [("IOI_01", 0), ("IOI_01", 1), ("IOI_01", 3), ("IOI_18", 0), ("IOI_18", 1), ("IOI_18", 3)]
    one: SynSample = samples[1]
    assert one.image.shard == "000001.tar"
    assert read_member(tmp_path, one.image) == f"{RENDER_PREFIX}/IOI_01/0001 png".encode()
    assert read_member(tmp_path, one.mask) == f"{RENDER_PREFIX}/IOI_01/0001 mask.png".encode()
    assert read_member(tmp_path, one.labels) == f"{RENDER_PREFIX}/IOI_01/0001 data.pyd".encode()


def test_bedlam_shards_index_each_clip_as_its_own_camera(tmp_path: Path) -> None:
    # BEDLAM names its frames <clip>_<frame> at 30 fps numbering and ships them unsorted.
    shard: Path = write_shard(
        tmp_path / "000000.tar",
        [member for clip, frame in [("seq_000001", 10), ("seq_000000", 5), ("seq_000000", 0)] for member in sample_members(f"{BEDLAM_PREFIX}/{clip}/{clip}_{frame:04d}")],
    )
    assert [(sample.camera, sample.frame) for sample in index_shards([shard])] == [("seq_000000", 0), ("seq_000000", 5), ("seq_000001", 10)]


def test_a_sample_missing_a_member_names_its_shard(tmp_path: Path) -> None:
    shard: Path = write_shard(tmp_path / "000003.tar", sample_members(f"{RENDER_PREFIX}/IOI_01/0000")[:2])
    with pytest.raises(ValueError, match=r"000003\.tar.*IOI_01/0000.*data\.pyd"):
        index_shards([shard])


def test_a_duplicate_member_names_its_shard(tmp_path: Path) -> None:
    first: Path = write_shard(tmp_path / "000000.tar", sample_members(f"{RENDER_PREFIX}/IOI_01/0000"))
    second: Path = write_shard(tmp_path / "000001.tar", sample_members(f"{RENDER_PREFIX}/IOI_01/0000")[:1])
    with pytest.raises(ValueError, match=r"000001\.tar.*IOI_01/0000\.png.*twice"):
        index_shards([first, second])


def render_person(*, pose: float = 0.0, trans: float = 0.0, cam_ext: np.ndarray | None = None) -> dict[str, object]:
    """One person's ``data.pyd`` record with the release's keys and dtypes (the fields the converter does not keep included)."""
    return {
        "imgname": "/tmp/bedlam_lab/x/png/seq/IOI_01/0000_00.png",
        "pose_world": np.full(165, pose),
        "shape": np.arange(16, dtype=np.float64),
        "trans_world": np.full(3, trans),
        "pose_cam": None,
        "trans_cam": None,
        "cam_int": np.array([[1790.0, 0.0, 1028.0], [0.0, 1788.0, 752.0], [0.0, 0.0, 1.0]]),
        "cam_ext": np.eye(4) if cam_ext is None else cam_ext,
        "gender": np.str_("neutral"),
        "joints3d": np.full((127, 3), pose + trans),
        "vertices3d": np.zeros((10475, 3)),
        "joints2d": np.zeros((29, 5, 127, 2)),
        "vertex_visibility": np.ones((10475, 1), dtype=np.int32),
        "motion_info": np.str_("unknown"),
        "sub": 0,
        "person_idx": 0,
    }


def pickled(people: dict[int, dict[str, object]]) -> bytes:
    return pickle.dumps(people, protocol=4)


def test_a_label_pickle_may_only_build_numpy_arrays() -> None:
    class Payload:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return (print, ("unpickled",))

    with pytest.raises(ValueError, match=r"IOI_01/0004.*builtins\.print"):
        read_render_view(pickle.dumps({0: Payload()}), "IOI_01/0004")


def test_render_views_assemble_each_person_over_the_sequence_frames() -> None:
    # Person 1 is seen only by IOI_02 at frame 3 and by nobody at frame 1: absent there, not invented.
    moved: np.ndarray = np.eye(4)
    moved[:3, 3] = [0.5, 0.0, 3.0]
    views: list[RenderView] = [
        RenderView("IOI_01", 0, read_render_view(pickled({0: render_person(pose=0.0), 1: render_person(pose=0.5)}), "IOI_01/0000")),
        RenderView("IOI_01", 1, read_render_view(pickled({0: render_person(pose=0.1)}), "IOI_01/0001")),
        RenderView("IOI_01", 3, read_render_view(pickled({0: render_person(pose=0.3)}), "IOI_01/0003")),
        RenderView("IOI_02", 0, read_render_view(pickled({0: render_person(pose=0.0, cam_ext=moved), 1: render_person(pose=0.5, cam_ext=moved)}), "IOI_02/0000")),
        RenderView("IOI_02", 3, read_render_view(pickled({1: render_person(pose=0.7, trans=1.0, cam_ext=moved)}), "IOI_02/0003")),
    ]
    labels: RenderLabels = assemble_render(views, {"IOI_01": (2056, 1504), "IOI_02": (1504, 2056)})
    np.testing.assert_array_equal(labels.frames, [0, 1, 3])
    first, second = labels.people
    np.testing.assert_array_equal(first.present, [True, True, True])
    np.testing.assert_array_equal(second.present, [True, False, True])
    np.testing.assert_array_equal(second.pose[:, 0], [0.5, 0.0, 0.7])
    np.testing.assert_array_equal(second.trans[2], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(first.betas, np.arange(16))
    assert (first.subject, first.gender, first.model, first.flat_hand_mean) == ("person_00", "neutral", "locked_head_neutral", True)
    assert np.isnan(labels.joints3d[1][1]).all() and (labels.joints3d[1][2] == 1.7).all()
    cameras: list[MammaCamera] = labels.cameras
    assert [(camera.name, camera.width, camera.height) for camera in cameras] == [("IOI_01", 2056, 1504), ("IOI_02", 1504, 2056)]
    np.testing.assert_array_equal(cameras[1].cam_T_world, moved)
    assert cameras[0].lens is None


def test_views_that_disagree_on_a_person_raise() -> None:
    views: list[RenderView] = [
        RenderView("IOI_01", 0, read_render_view(pickled({0: render_person(pose=0.0)}), "IOI_01/0000")),
        RenderView("IOI_02", 0, read_render_view(pickled({0: render_person(pose=0.2)}), "IOI_02/0000")),
    ]
    with pytest.raises(ValueError, match=r"person 0.*frame 0.*IOI_02"):
        assemble_render(views, {"IOI_01": (2056, 1504), "IOI_02": (2056, 1504)})


def test_a_render_camera_that_moves_raises() -> None:
    moved: np.ndarray = np.eye(4)
    moved[0, 3] = 0.01
    views: list[RenderView] = [
        RenderView("IOI_01", 0, read_render_view(pickled({0: render_person()}), "IOI_01/0000")),
        RenderView("IOI_01", 1, read_render_view(pickled({0: render_person(cam_ext=moved)}), "IOI_01/0001")),
    ]
    with pytest.raises(ValueError, match=r"IOI_01.*frame 1.*moves"):
        assemble_render(views, {"IOI_01": (2056, 1504)})
