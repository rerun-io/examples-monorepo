"""MammaSyn shard readers against hand-built WebDataset tars laid out as the release ships them."""

import io
import json
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

from dataforge.datasets import mamma_syn_source
from dataforge.datasets.mamma import MammaConfig
from dataforge.datasets.mamma_body import PosedPerson
from dataforge.datasets.mamma_source import MammaCamera, PersonFit
from dataforge.datasets.mamma_syn import MammaSynDataset, check_joints
from dataforge.datasets.mamma_syn_source import (
    BEDLAM_DATASET,
    SYN_DIR,
    BedlamLabels,
    BedlamView,
    RenderLabels,
    RenderPersonLabel,
    RenderView,
    ShardMember,
    ShardReader,
    SynSample,
    assemble_bedlam,
    assemble_render,
    copy_span,
    discover_renders,
    index_shards,
    read_bedlam_view,
    read_members,
    read_render_view,
    sequence_samples,
    stage_shards,
)

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


def read_member(directory: Path, member: ShardMember) -> bytes:
    with ShardReader(directory) as reader:
        return reader.read(member)


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


def test_staged_shards_are_identical_copies_the_cached_index_still_describes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A sequence is copied to local disk before its convert reads it: spans of a few bytes stitch back into the same shards.
    monkeypatch.setattr(mamma_syn_source, "READ_SPAN", 7)
    nas: Path = tmp_path / "nas"
    nas.mkdir()
    first: Path = write_shard(nas / "000000.tar", [member for frame in (0, 1) for member in sample_members(f"{RENDER_PREFIX}/IOI_01/{frame:04d}")])
    second: Path = write_shard(nas / "000001.tar", sample_members(f"{RENDER_PREFIX}/IOI_01/0003"))
    indexed: list[SynSample] = sequence_samples([first, second], tmp_path / "index", "sequence")
    copies: list[Path] = stage_shards([first, second], tmp_path / "stage")
    assert [copy.read_bytes() for copy in copies] == [first.read_bytes(), second.read_bytes()]
    assert sequence_samples(copies, tmp_path / "index", "sequence") == indexed
    assert read_member(tmp_path / "stage", indexed[2].labels) == f"{RENDER_PREFIX}/IOI_01/0003 data.pyd".encode()


def test_members_read_ahead_in_runs_are_the_members_read_one_by_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A BEDLAM clip's samples lie in a few runs of its batch's shards, between other clips' samples; adjacent members come in one read.
    # These tiny members sit ~2 kB apart (long-name tar headers); another clip's sample between two is a gap of ~8 kB.
    monkeypatch.setattr(mamma_syn_source, "READ_GAP", 4096)
    order: list[tuple[str, int]] = [("seq_000000", 0), ("seq_000000", 5), ("seq_000001", 0), ("seq_000000", 10)]
    write_shard(tmp_path / "000000.tar", [member for clip, frame in order for member in sample_members(f"{BEDLAM_PREFIX}/{clip}/{clip}_{frame:04d}")])
    write_shard(tmp_path / "000001.tar", sample_members(f"{BEDLAM_PREFIX}/seq_000000/seq_000000_0015"))
    clip: list[SynSample] = [sample for sample in index_shards(sorted(tmp_path.glob("*.tar"))) if sample.camera == "seq_000000"]
    members: list[ShardMember] = [member for sample in clip for member in (sample.labels, sample.image, sample.mask)]
    reads: list[ShardMember] = []
    original = ShardReader.read

    def counted(reader: ShardReader, member: ShardMember) -> bytes:
        reads.append(member)
        return original(reader, member)

    monkeypatch.setattr(ShardReader, "read", counted)
    read: dict[tuple[str, int], bytes] = read_members(tmp_path, members)
    monkeypatch.setattr(ShardReader, "read", original)
    assert len(read) == len(members) == 12
    assert sorted(member.shard for member in reads) == ["000000.tar", "000000.tar", "000001.tar"]  # frames 0 and 5 in one read, 10, 15
    with ShardReader(tmp_path, read) as cached:
        assert [cached.read(member) for member in members] == [read_member(tmp_path, member) for member in members]


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


VERTICES: int = 10475


def vertex_mask(vertices: list[int], dtype: type = bool) -> np.ndarray:
    """A per-vertex column, 1 at ``vertices``."""
    mask: np.ndarray = np.zeros(VERTICES, dtype=dtype)
    mask[vertices] = 1
    return mask


def render_person(
    *,
    pose: float = 0.0,
    trans: float = 0.0,
    cam_ext: np.ndarray | None = None,
    floor: list[int] | None = None,
    distance: np.ndarray | None = None,
    seen: list[int] | None = None,
    rate: float = 1.0,
) -> dict[str, object]:
    """One person's ``data.pyd`` record with the release's keys and dtypes (the fields the converter does not keep included).
    ``floor``: the vertices on the floor; ``distance``: the per-vertex distance to the other person (an empty list without);
    ``seen``: the vertices this view sees (all without)."""
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
        "vertices2d": np.zeros((VERTICES, 2)),
        "vertex_visibility": (np.ones(VERTICES, dtype=np.int32) if seen is None else vertex_mask(seen, np.int32))[:, None],
        "person_visitiblity_rate": np.float64(rate),
        "sdf_vertices": [] if distance is None else distance.astype(np.float32)[:, None],
        "floor_contact_mask": vertex_mask(floor or []),
        "camera_normals": np.zeros((VERTICES, 3)),
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


def test_render_views_carry_each_persons_contact_and_what_each_view_sees() -> None:
    # Floor contact and the distance to the other person belong to the person at a frame (every view ships the same);
    # visibility belongs to the view. Frame 1 ships no distance, and person 1 is absent there.
    distance: np.ndarray = np.linspace(-0.01, 0.1, VERTICES)
    views: list[RenderView] = [
        RenderView("IOI_01", 0, read_render_view(pickled({0: render_person(floor=[3, 4], distance=distance, seen=[0, 1]), 1: render_person(pose=0.5, distance=-distance)}), "IOI_01/0000")),
        RenderView("IOI_02", 0, read_render_view(pickled({0: render_person(floor=[3, 4], distance=distance, seen=[2], rate=0.4), 1: render_person(pose=0.5, distance=-distance)}), "IOI_02/0000")),
        RenderView("IOI_01", 1, read_render_view(pickled({0: render_person(pose=0.1, floor=[9])}), "IOI_01/0001")),
    ]
    labels: RenderLabels = assemble_render(views, {"IOI_01": (2056, 1504), "IOI_02": (2056, 1504)})
    first, second = labels.contacts
    assert (first.partner, second.partner) == (1, 0)
    assert [np.flatnonzero(row).tolist() for row in first.floor] == [[3, 4], [9]]
    assert not second.floor.any()
    assert first.partner_distance_m is not None and second.partner_distance_m is not None
    np.testing.assert_array_equal(first.partner_distance_m[0], distance.astype(np.float32))
    np.testing.assert_array_equal(second.partner_distance_m[0], -distance.astype(np.float32))
    assert np.isnan(first.partner_distance_m[1]).all()
    seen: RenderPersonLabel = views[1].people[0]
    assert (np.flatnonzero(seen.vertex_visibility).tolist(), seen.person_visitiblity_rate) == ([2], 0.4)
    assert np.flatnonzero(views[0].people[0].vertex_visibility).tolist() == [0, 1]


def test_views_that_disagree_on_a_persons_floor_contact_raise() -> None:
    views: list[RenderView] = [
        RenderView("IOI_01", 0, read_render_view(pickled({0: render_person(floor=[3])}), "IOI_01/0000")),
        RenderView("IOI_02", 0, read_render_view(pickled({0: render_person(floor=[4])}), "IOI_02/0000")),
    ]
    with pytest.raises(ValueError, match=r"person 0.*disagree at frame 0.*IOI_02"):
        assemble_render(views, {"IOI_01": (2056, 1504), "IOI_02": (2056, 1504)})


def test_a_distance_to_the_other_person_in_a_scene_of_three_raises() -> None:
    # The release does not say whom a distance is to: only a two-person scene has one other person.
    distance: np.ndarray = np.zeros(VERTICES)
    people: dict[int, dict[str, object]] = {key: render_person(pose=0.1 * key, distance=distance) for key in (0, 1, 2)}
    views: list[RenderView] = [RenderView("IOI_01", 0, read_render_view(pickled(people), "IOI_01/0000"))]
    with pytest.raises(ValueError, match=r"3 people.*distance"):
        assemble_render(views, {"IOI_01": (2056, 1504)})


def test_a_vertex_visibility_other_than_0_or_1_names_its_sample() -> None:
    person: dict[str, object] = render_person()
    person["vertex_visibility"] = np.full((VERTICES, 1), 0.5)
    with pytest.raises(ValueError, match=r"IOI_01/0004.*0 and 1"):
        read_render_view(pickled({0: person}), "IOI_01/0004")


def test_a_render_camera_that_moves_raises() -> None:
    moved: np.ndarray = np.eye(4)
    moved[0, 3] = 0.01
    views: list[RenderView] = [
        RenderView("IOI_01", 0, read_render_view(pickled({0: render_person()}), "IOI_01/0000")),
        RenderView("IOI_01", 1, read_render_view(pickled({0: render_person(cam_ext=moved)}), "IOI_01/0001")),
    ]
    with pytest.raises(ValueError, match=r"IOI_01.*frame 1.*moves"):
        assemble_render(views, {"IOI_01": (2056, 1504)})


def bedlam_person(*, offset: float, frame: int, focal: float = 1010.0, floor: list[int] | None = None, seen: list[int] | None = None) -> dict[str, object]:
    """One person of a BEDLAM view: float32 pose and 11 betas, its own world (cam_ext), joints in the camera frame; float64
    visibility, and no distance to anyone (BEDLAM ships none)."""
    cam_ext: np.ndarray = np.eye(4)
    cam_ext[:3, 3] = [offset, 0.0, 4.0 + 0.1 * frame]  # each person's world sits elsewhere; the orbit moves the camera
    return {
        "pose_world": np.full(165, 0.01 * frame, dtype=np.float32),
        "shape": np.arange(11, dtype=np.float32),
        "trans_world": np.array([0.0, -0.6, 0.0]),
        "pose_cam": np.zeros(165, dtype=np.float32),
        "trans_cam": np.zeros(3),
        "cam_int": np.array([[focal, 0.0, 640.0], [0.0, focal, 360.0], [0.0, 0.0, 1.0]]),
        "cam_ext": cam_ext,
        "gender": "female",
        "joints3d": np.full((127, 3), offset + frame),
        "vertices2d": np.zeros((10475, 2), dtype=np.float32),
        "motion_info": np.array(["kick", "29", "amass.npz", "120.0", "0", "526"]),
        "sub": "rp_christine_posed_013",
        "person_idx": "00",
        "race": "asian",
        "floor_contact_mask": vertex_mask(floor or []),
        "camera_normals": np.zeros((VERTICES, 3)),
        "vertex_visibility": (np.ones(VERTICES) if seen is None else vertex_mask(seen, np.float64))[:, None],
        "person_visitiblity_rate": np.float64(1.0),
    }


def test_bedlam_views_keep_each_persons_camera_and_the_zoom() -> None:
    views: list[BedlamView] = [
        BedlamView(0, read_bedlam_view(pickled({0: bedlam_person(offset=0.0, frame=0), 2: bedlam_person(offset=1.5, frame=0)}), "seq_000000/0000")),
        BedlamView(10, read_bedlam_view(pickled({0: bedlam_person(offset=0.0, frame=10, focal=1500.0)}), "seq_000000/0010")),
        BedlamView(5, read_bedlam_view(pickled({0: bedlam_person(offset=0.0, frame=5, focal=1200.0), 2: bedlam_person(offset=1.5, frame=5, focal=1200.0)}), "seq_000000/0005")),
    ]
    labels: BedlamLabels = assemble_bedlam(views)
    np.testing.assert_array_equal(labels.frames, [0, 5, 10])
    np.testing.assert_array_equal(labels.K[:, 0, 0], [1010.0, 1200.0, 1500.0])
    assert [person.index for person in labels.people] == [0, 2]
    first, second = labels.people
    assert (first.model, first.gender, first.flat_hand_mean, len(first.betas)) == ("v1_1_neutral", "female", True, 11)
    np.testing.assert_array_equal(second.present, [True, True, False])
    np.testing.assert_allclose(first.pose[:, 0], [0.0, 0.05, 0.1], rtol=1e-6)
    np.testing.assert_array_equal(labels.cam_T_body[1][1][:3, 3], [1.5, 0.0, 4.5])
    assert np.isnan(labels.cam_T_body[1][2]).all() and np.isnan(labels.joints3d[1][2]).all()
    assert (labels.joints3d[0][2] == 10.0).all()


def test_bedlam_views_carry_each_persons_floor_contact_and_motion_source() -> None:
    views: list[BedlamView] = [
        BedlamView(0, read_bedlam_view(pickled({0: bedlam_person(offset=0.0, frame=0, floor=[7], seen=[5])}), "seq_000000/0000")),
        BedlamView(5, read_bedlam_view(pickled({0: bedlam_person(offset=0.0, frame=5)}), "seq_000000/0005")),
    ]
    labels: BedlamLabels = assemble_bedlam(views)
    (contact,) = labels.contacts
    assert [np.flatnonzero(row).tolist() for row in contact.floor] == [[7], []]
    assert (contact.partner, contact.partner_distance_m) == (None, None)
    assert labels.motions == [["kick", "29", "amass.npz", "120.0", "0", "526"]]
    assert np.flatnonzero(views[0].people[0].vertex_visibility).tolist() == [5]


def test_people_in_one_bedlam_view_that_disagree_on_the_intrinsics_raise() -> None:
    views: list[BedlamView] = [BedlamView(0, read_bedlam_view(pickled({0: bedlam_person(offset=0.0, frame=0), 1: bedlam_person(offset=1.0, frame=0, focal=999.0)}), "x"))]
    with pytest.raises(ValueError, match=r"frame 0.*person 1.*intrinsics"):
        assemble_bedlam(views)


def test_a_short_read_while_staging_fails_instead_of_keeping_a_padded_copy(tmp_path: Path) -> None:
    # The copy is pre-sized, so a span cut short (a source truncated mid-copy) would otherwise pass the size and mtime check.
    (tmp_path / "000000.tar").write_bytes(b"abcd")
    (tmp_path / "copy").write_bytes(bytes(8))
    with pytest.raises(ValueError, match=r"000000.tar: 4 of 8 bytes at 0"):
        copy_span(tmp_path / "000000.tar", tmp_path / "copy", 0, 8)


def test_each_run_stages_in_a_directory_of_its_own_and_removes_those_of_runs_that_died(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    parent: Path = tmp_path / "out" / "work" / "mamma-syn-stage"
    dead: Path = parent / "999999999" / "mamma-syn__dataset__sequence"  # a pid no process has
    dead.mkdir(parents=True)
    alive: Path = parent / str(os.getppid())
    alive.mkdir()
    dataset = MammaConfig(family="syn", root=tmp_path, index_root=tmp_path / "index").setup()
    assert isinstance(dataset, MammaSynDataset)
    assert dataset.stage_root == parent / str(os.getpid()) and dataset.stage_root.is_dir()
    assert not dead.parent.exists() and alive.is_dir()


def test_a_bedlam_batch_whose_shards_miss_its_metadata_count_fails_discovery(tmp_path: Path) -> None:
    batch: Path = tmp_path / SYN_DIR / BEDLAM_DATASET / "20221010_3-10_500_batch01hand_zoom_suburb_d"
    batch.mkdir(parents=True)
    write_shard(batch / "000000.tar", sample_members(f"{BEDLAM_PREFIX}/seq_000000/seq_000000_0000"))
    (batch / "metadata.json").write_text(json.dumps({"format": "webdataset", "num_samples": 2}))
    dataset = MammaConfig(family="syn-bedlam", root=tmp_path, index_root=tmp_path / "index").setup()
    with pytest.raises(ValueError, match=r"1 samples in the shards, metadata.json declares 2"):
        dataset.discover()


def test_the_joints_check_fails_a_present_row_with_non_finite_joints() -> None:
    # Absent rows are NaN by design; a NaN in a row the labels name is a broken label, not a match.
    present: np.ndarray = np.array([True, True, False])
    person = PersonFit(
        subject="person_03", index=3, gender="neutral", model="locked_head_neutral", flat_hand_mean=True,
        pose=np.zeros((3, 165)), trans=np.zeros((3, 3)), betas=np.zeros(16), v_template=None, present=present,
    )
    joints: np.ndarray = np.zeros((3, 144, 3), dtype=np.float32)
    joints[2] = np.nan
    posed = PosedPerson(
        index=3, coco133=np.zeros((3, 133, 3), dtype=np.float32), joints=joints, mesh_rows=np.arange(3), mesh_present=present,
        vertices=np.zeros((2, 10475, 3), dtype=np.float32), faces=np.zeros((20908, 3), dtype=np.uint32),
    )
    shipped: np.ndarray = np.zeros((3, 127, 3))
    shipped[2] = np.nan
    frames: np.ndarray = np.array([0, 10, 20])
    assert check_joints([person], [posed], [shipped], frames) == 0.0
    shipped[1, 5] = np.nan
    with pytest.raises(ValueError, match=r"person_03: frame 10"):
        check_joints([person], [posed], [shipped], frames)


def test_a_render_sequence_the_release_ships_empty_is_skipped_and_named(tmp_path: Path) -> None:
    # Two release directories hold one empty tar and a metadata.json declaring 0 samples: nothing to convert, not a failure.
    dataset: Path = tmp_path / SYN_DIR / "MammaSyn-Interactions" / "latindance10_1_NC_entire_dataset_00_contact"
    full: Path = dataset / "be_dkrWAwL7TcOZ_seq_000002"
    empty: Path = dataset / "be_dkrWAwL7TcOZ_seq_000003"
    for directory, members, count in ((full, sample_members(f"{RENDER_PREFIX}/IOI_01/0000"), 1), (empty, [], 0)):
        directory.mkdir(parents=True)
        write_shard(directory / "000000.tar", members)
        (directory / "metadata.json").write_text(json.dumps({"format": "webdataset", "num_samples": count, "count_crop": 0}))
    found, skipped = discover_renders(tmp_path)
    assert [sequence.directory for sequence in found] == [full]
    assert skipped == [(empty, "empty: metadata.json declares no samples")]
