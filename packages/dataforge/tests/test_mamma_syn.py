"""MammaSyn end to end on a synthetic render sequence laid out as the release ships it (real SMPL-X model, WebDataset shards)."""

import io
import json
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest
import smplx
import torch
from conftest import column_rows, raw_asset, read_back, read_chunks, recording_properties, static_values
from PIL import Image

from dataforge import paths, schema
from dataforge.clocks import frame_times
from dataforge.datasets.mamma import MammaConfig
from dataforge.datasets.mamma_body import MODEL_FILES
from dataforge.datasets.mamma_source import SmplxModel
from dataforge.datasets.mamma_syn import MammaBedlamDataset, MammaSynDataset

pytestmark = pytest.mark.integration

MODEL_ROOT: Path = Path(os.environ.get("DATAFORGE_MAMMA_MODEL_ROOT", str(paths.raw_root() / "mamma-body-models")))
DATASET: str = "testhands_2-6_NC_200_00"
SEQUENCE: str = "be_test_seq_000001"
FRAMES: tuple[int, ...] = (0, 1, 3)
"""Frame 2 is missing in every camera, as in the sampled Hands sequence: a gap on the clock, not a renumbering."""
CAMERAS: dict[str, tuple[int, int]] = {"IOI_01": (320, 240), "IOI_02": (240, 320)}
ABSENT: tuple[int, int] = (1, 1)
"""(person, frame): nobody's view labels person 1 at frame 1."""


def body_params(person: int, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose: np.ndarray = np.zeros(165)
    pose[0] = np.pi
    pose[3 + 3 * person] = 0.1 * frame  # each person and frame its own pose
    trans: np.ndarray = np.array([0.6 * person - 0.3, 0.0, 0.0])
    return pose, np.full(16, 0.1 * person), trans


RENDER_MODELS: tuple[SmplxModel, SmplxModel] = ("locked_head_neutral", "v1_1_neutral")
"""The model each person was rendered with: MAMMA's datasets use one or the other (Interhand the locked head, Inter-X v1.1)."""


def shipped_joints(person: int, pose: np.ndarray, betas: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """The render's joints3d: the person's model's 127 joints (no face contour), flat hand mean, world frame."""
    model = smplx.create(str(MODEL_FILES[RENDER_MODELS[person]].local(MODEL_ROOT)), model_type="smplx", use_pca=False, flat_hand_mean=True, num_betas=16)
    tensor: torch.Tensor = torch.from_numpy(pose[None].astype(np.float32))
    with torch.no_grad():
        output = model(
            betas=torch.from_numpy(betas[None].astype(np.float32)),
            global_orient=tensor[:, :3],
            body_pose=tensor[:, 3:66],
            jaw_pose=tensor[:, 66:69],
            leye_pose=tensor[:, 69:72],
            reye_pose=tensor[:, 72:75],
            left_hand_pose=tensor[:, 75:120],
            right_hand_pose=tensor[:, 120:165],
            transl=torch.from_numpy(trans[None].astype(np.float32)),
        )
    return output.joints[0, :127].numpy().astype(np.float64)


def camera_matrices(name: str) -> tuple[np.ndarray, np.ndarray]:
    width, height = CAMERAS[name]
    cam_ext: np.ndarray = np.eye(4)
    cam_ext[:3, 3] = [0.0, 0.0, 4.0] if name == "IOI_01" else [0.2, 0.0, 5.0]
    return np.array([[0.9 * width, 0.0, width / 2], [0.0, 0.9 * width, height / 2], [0.0, 0.0, 1.0]]), cam_ext


def png(array: np.ndarray) -> bytes:
    """RGBA for an h x w x 4 array, 8-bit gray for an h x w one."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def write_sequence(root: Path, *, joint_error_m: float = 0.0) -> Path:
    """One render sequence in two shards; IOI_01 straddles them, as cameras do in the release."""
    directory: Path = root / "MammaSyn" / "MammaSyn-Hands" / DATASET / SEQUENCE
    directory.mkdir(parents=True)
    joints: dict[tuple[int, int], np.ndarray] = {
        (person, frame): shipped_joints(person, *body_params(person, frame)) for person in (0, 1) for frame in FRAMES
    }
    members: list[tuple[str, bytes]] = []
    for name, (width, height) in CAMERAS.items():
        cam_int, cam_ext = camera_matrices(name)
        for frame in FRAMES:
            key: str = f"/tmp/bedlam_lab_20251031_191434/{DATASET}/png/{SEQUENCE}/{name}/{frame:04d}"
            people: dict[int, dict[str, object]] = {}
            mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
            for person in (0, 1):
                if (person, frame) == ABSENT:
                    continue
                pose, betas, trans = body_params(person, frame)
                people[person] = {
                    "pose_world": pose,
                    "shape": betas,
                    "trans_world": trans,
                    "pose_cam": None,
                    "trans_cam": None,
                    "cam_int": cam_int,
                    "cam_ext": cam_ext,
                    "gender": np.str_("neutral"),
                    "joints3d": joints[(person, frame)] + joint_error_m,
                    "vertices2d": np.zeros((10475, 2)),
                    "motion_info": np.str_("unknown"),
                    "sub": 0,
                    "person_idx": person,
                }
                mask[:, person * width // 2 : (person + 1) * width // 2] = person + 1
            rgba: np.ndarray = np.full((height, width, 4), 255, dtype=np.uint8)
            rgba[..., 0] = 40 * frame
            members += [(f"{key}.png", png(rgba)), (f"{key}.mask.png", png(mask)), (f"{key}.data.pyd", pickle.dumps(people, protocol=4))]
    split: int = 3 * 2  # two samples of IOI_01 in the first shard
    for index, chunk in enumerate((members[:split], members[split:])):
        with tarfile.open(directory / f"{index:06d}.tar", "w") as tar:
            for name, payload in chunk:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    (directory / "metadata.json").write_text(json.dumps({"format": "webdataset", "num_samples": len(members) // 3, "count_crop": 11}))
    return directory


@pytest.fixture
def models() -> Path:
    for model in MODEL_FILES.values():
        raw_asset("MAMMA SMPL-X model (dataforge-download mamma)", model.local(MODEL_ROOT))
    return MODEL_ROOT


def syn_dataset(root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_root: Path) -> MammaSynDataset:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family="syn", root=root, model_root=model_root, index_root=tmp_path / "index").setup()
    assert isinstance(dataset, MammaSynDataset)
    return dataset


def test_a_render_sequence_converts_onto_its_source_frames(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path) -> None:
    del nvenc_ffmpeg  # NVENC must be present; the converter resolves ffmpeg itself
    write_sequence(tmp_path / "raw")
    dataset: MammaSynDataset = syn_dataset(tmp_path / "raw", tmp_path, monkeypatch, models)
    ((identity, sequence),) = dataset.discover()
    assert identity.recording_id == f"mamma-syn__{DATASET}__{SEQUENCE}"
    dataset.prefetch(identity, sequence, force=False)  # the runner indexes the next sequence's shards while one converts
    assert (tmp_path / "index" / f"{identity.sequence_key}.json").is_file()
    dataset.convert(identity, sequence, force=False)
    targets: dict[str, Path] = {layer: target for layer, target in dataset.targets(identity).items() if target.is_file()}
    assert sorted(targets) == sorted(["base", "body_pose", "body_mesh", "projections", "masks"])
    times: list[int] = frame_times(np.array(FRAMES, dtype=np.int64), 6).tolist()

    base = read_back(targets["base"])
    for rig, (name, (width, height)) in enumerate(CAMERAS.items(), start=1):
        column: str = f"{schema.video_path(rig, 0)}:VideoStream:sample"
        assert column_rows(base, column)[schema.TIMELINE].cast("int64").to_pylist() == times
        by_frame = base.reader(index=schema.FRAME_INDEX).to_arrow_table().select([schema.FRAME_INDEX, column]).drop_null()
        assert sorted(by_frame[schema.FRAME_INDEX].to_pylist()) == list(FRAMES)
        values = static_values(base, schema.cam_path(rig, 0))
        assert (values["name"], values["source_width"], values["source_height"], values["camera_model"]) == (name, width, height, "pinhole")
    capture = recording_properties(base, "capture")
    assert (capture["num_frames"], capture["fps"]) == (len(FRAMES), 6)
    episode = recording_properties(base, "episode")
    assert (episode["group"], episode["collection"], episode["body_source"], episode["num_people"]) == ("Hands", DATASET, "synthetic_gt", 2)

    pose = read_back(targets["body_pose"])
    for person in (0, 1):
        rows = column_rows(pose, f"{schema.person_body_path(person, 'smplx')}:pose")
        assert rows[schema.TIMELINE].cast("int64").to_pylist() == times
        shipped: np.ndarray = np.array([body_params(person, frame)[0] for frame in FRAMES])
        stored: np.ndarray = np.array([np.full(165, np.nan) if row is None else row for row in rows.column(1).to_pylist()])[:, 0]
        if person == 1:
            assert np.isnan(stored[1]).all()  # absent at frame 1
            shipped, stored = shipped[[0, 2]], stored[[0, 2]]
        np.testing.assert_array_equal(stored, shipped)
        values = static_values(pose, schema.person_path(person))
        assert values["source"] == "synthetic_gt"
        assert values["model"] == MODEL_FILES[RENDER_MODELS[person]].path  # the model whose forward gives the shipped joints back

    mesh = read_back(targets["body_mesh"])
    rows = column_rows(mesh, f"{schema.person_body_path(1, 'mesh')}:Mesh3D:vertex_positions")
    assert rows[schema.TIMELINE].cast("int64").to_pylist() == times  # every frame at 6 Hz
    assert [len(row) for row in rows.column(1).to_pylist()[0:3]] in ([10475, 0, 10475], [[10475], [0], [10475]])

    projected = {str(chunk.entity_path) for chunk in read_chunks(targets["projections"]) if not chunk.is_static}
    assert projected == {schema.person_projection_path(rig, 0, person) for rig in (1, 2) for person in (0, 1)}

    masks = read_back(targets["masks"])
    for rig, (width, height) in enumerate(CAMERAS.values(), start=1):
        small: tuple[int, int] = (width // 4, height // 4)
        for person in (0, 1):
            rows = column_rows(masks, f"{schema.person_mask_path(rig, 0, person)}:EncodedImage:blob")
            assert rows[schema.TIMELINE].cast("int64").to_pylist() == times
            ids: list[np.ndarray] = [np.asarray(Image.open(io.BytesIO(bytes(blob[0])))) for blob in rows.column(1).to_pylist()]
            expected: np.ndarray = np.zeros(small[::-1], dtype=np.uint8)
            expected[:, person * small[0] // 2 : (person + 1) * small[0] // 2] = 1
            for frame, image in zip(FRAMES, ids, strict=True):
                np.testing.assert_array_equal(image, np.zeros_like(expected) if (person, frame) == ABSENT else expected)


def test_parameters_that_do_not_reproduce_the_shipped_joints_fail_the_sequence(
    tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path
) -> None:
    del nvenc_ffmpeg
    write_sequence(tmp_path / "raw", joint_error_m=0.01)
    dataset: MammaSynDataset = syn_dataset(tmp_path / "raw", tmp_path, monkeypatch, models)
    ((identity, sequence),) = dataset.discover()
    with pytest.raises(ValueError, match=r"person_00.*joints3d.*10\.0 mm"):
        dataset.convert(identity, sequence, force=False)


BATCH: str = "20221010_3-10_500_batch01hand_zoom_suburb_d"
CLIPS: dict[str, tuple[int, ...]] = {"seq_000000": (10, 0, 5), "seq_000001": (0, 5)}
"""Frames in shard order: BEDLAM ships a clip's samples unsorted, numbered at 30 fps, every 5th."""
BEDLAM_SIZE: tuple[int, int] = (320, 240)


def bedlam_forward(frame: int, person: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A body in its own world: pose, betas, transl, and the v1.1 neutral's vertices and 127 joints (flat hand mean, 11 betas)."""
    pose: np.ndarray = np.zeros(165, dtype=np.float32)
    pose[0] = np.pi
    pose[3 + 3 * person] = 0.02 * frame
    betas: np.ndarray = np.full(11, 0.2 * person, dtype=np.float32)
    trans: np.ndarray = np.array([0.0, 0.0, 0.0])
    model = smplx.create(str(MODEL_FILES["v1_1_neutral"].local(MODEL_ROOT)), model_type="smplx", use_pca=False, flat_hand_mean=True, num_betas=11)
    tensor: torch.Tensor = torch.from_numpy(pose[None])
    with torch.no_grad():
        output = model(
            betas=torch.from_numpy(betas[None]),
            global_orient=tensor[:, :3],
            body_pose=tensor[:, 3:66],
            jaw_pose=tensor[:, 66:69],
            leye_pose=tensor[:, 69:72],
            reye_pose=tensor[:, 72:75],
            left_hand_pose=tensor[:, 75:120],
            right_hand_pose=tensor[:, 120:165],
            transl=torch.from_numpy(trans[None].astype(np.float32)),
        )
    return pose, betas, trans, output.vertices[0].numpy().astype(np.float64), output.joints[0, :127].numpy().astype(np.float64)


def bedlam_camera(frame: int, person: int) -> tuple[np.ndarray, np.ndarray]:
    """The zoom: K changes every frame; each person's world sits at its own place in front of the camera."""
    focal: float = 300.0 + 10.0 * frame
    cam_ext: np.ndarray = np.eye(4)
    cam_ext[:3, 3] = [0.8 * person - 0.4, 0.0, 5.0]
    return np.array([[focal, 0.0, 160.0], [0.0, focal, 120.0], [0.0, 0.0, 1.0]]), cam_ext


def write_bedlam_batch(root: Path) -> Path:
    directory: Path = root / "MammaSyn" / "MammaSyn-Singles" / "BEDLAM_MASKS_WD" / BATCH
    directory.mkdir(parents=True)
    members: list[tuple[str, bytes]] = []
    width, height = BEDLAM_SIZE
    for clip, frames in CLIPS.items():
        for frame in frames:
            people: dict[int, dict[str, object]] = {}
            mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
            for person in (0, 1):
                mask[: height // 2, person * width // 2 : (person + 1) * width // 2] = person + 1  # the mask shows both, always
                if (clip, frame, person) == ("seq_000000", 10, 1):
                    continue  # BEDLAM's labels leave out a person its mask still has
                pose, betas, trans, _, joints = bedlam_forward(frame, person)
                cam_int, cam_ext = bedlam_camera(frame, person)
                people[person] = {
                    "pose_world": pose,
                    "shape": betas,
                    "trans_world": trans,
                    "cam_int": cam_int,
                    "cam_ext": cam_ext,
                    "gender": "female" if person == 0 else "male",
                    "joints3d": joints @ cam_ext[:3, :3].T + cam_ext[:3, 3],
                    "person_idx": f"{person:02d}",
                }
            key: str = f"/ps/project/bedlam/public/b0_all/{BATCH}/png/{clip}/{clip}_{frame:04d}"
            rgba: np.ndarray = np.full((height, width, 4), 255, dtype=np.uint8)
            members += [(f"{key}.data.pyd", pickle.dumps(people, protocol=4)), (f"{key}.png", png(rgba)), (f"{key}.mask.png", png(mask))]
    with tarfile.open(directory / "000000.tar", "w") as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    (directory / "metadata.json").write_text(json.dumps({"format": "webdataset", "num_samples": len(members) // 3, "count_crop": 9}))
    return directory


def test_a_bedlam_clip_converts_in_its_camera_frame(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path) -> None:
    del nvenc_ffmpeg
    write_bedlam_batch(tmp_path / "raw")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family="syn-bedlam", root=tmp_path / "raw", model_root=models, index_root=tmp_path / "index").setup()
    assert isinstance(dataset, MammaBedlamDataset)
    pairs = dataset.discover()
    assert [identity.recording_id for identity, _ in pairs] == [f"mamma-syn-bedlam__{BATCH}__seq_000000", f"mamma-syn-bedlam__{BATCH}__seq_000001"]
    identity, clip = pairs[0]
    dataset.convert(identity, clip, force=False)
    targets: dict[str, Path] = {layer: target for layer, target in dataset.targets(identity).items() if target.is_file()}
    assert sorted(targets) == sorted(["base", "body_pose", "body_mesh", "projections", "masks"])
    frames: list[int] = [0, 5, 10]
    times: list[int] = frame_times(np.array(frames, dtype=np.int64), 30).tolist()

    base = read_back(targets["base"])
    assert column_rows(base, f"{schema.video_path(1, 0)}:VideoStream:sample")[schema.TIMELINE].cast("int64").to_pylist() == times
    intrinsics = column_rows(base, f"{schema.pinhole_path(1, 0)}:Pinhole:image_from_camera")
    assert [row[0][0] for row in intrinsics.column(1).to_pylist()] == [300.0, 350.0, 400.0]  # one K per row; fx leads either order

    mesh = read_back(targets["body_mesh"])
    rows = column_rows(mesh, f"{schema.person_body_path(0, 'mesh')}:Mesh3D:vertex_positions")
    first: np.ndarray = np.array(rows.column(1).to_pylist()[0])
    _, _, _, vertices, _ = bedlam_forward(0, 0)
    _, cam_ext = bedlam_camera(0, 0)
    np.testing.assert_allclose(first, vertices @ cam_ext[:3, :3].T + cam_ext[:3, 3], atol=1e-4)  # posed in the camera frame
    absent = column_rows(mesh, f"{schema.person_body_path(1, 'mesh')}:Mesh3D:vertex_positions").column(1).to_pylist()
    assert len(absent[2]) == 0

    pose = read_back(targets["body_pose"])
    keypoints = np.array(column_rows(pose, f"{schema.person_coco133_xyz_path(0)}:Points3D:positions").column(1).to_pylist())
    projected = np.array(column_rows(read_back(targets["projections"]), f"{schema.person_projection_path(1, 0, 0)}:Points2D:positions").column(1).to_pylist())
    for row, frame in enumerate(frames):
        cam_int, _ = bedlam_camera(frame, 0)
        expected: np.ndarray = keypoints[row] @ cam_int.T
        np.testing.assert_allclose(projected[row], expected[:, :2] / expected[:, 2:], rtol=1e-4)  # each frame through its own K

    masks = read_back(targets["masks"])
    for person in (0, 1):
        blobs = column_rows(masks, f"{schema.person_mask_path(1, 0, person)}:EncodedImage:blob").column(1).to_pylist()
        assert len(blobs) == 3 and np.asarray(Image.open(io.BytesIO(bytes(blobs[2][0])))).any()  # person 1's mask stays at frame 10
