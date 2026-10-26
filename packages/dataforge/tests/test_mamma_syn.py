"""MammaSyn end to end on a synthetic render sequence laid out as the release ships it (real SMPL-X model, WebDataset shards)."""

import io
import json
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun.blueprint as rrb
import smplx
import torch
from conftest import column_rows, raw_asset, read_back, read_chunks, recording_properties, static_values
from PIL import Image
from test_mamma_layout import scene

from dataforge import paths, schema
from dataforge.clocks import frame_times
from dataforge.datasets.mamma import MammaConfig
from dataforge.datasets.mamma_body import BODY_FILES, MODEL_FILES, landmark_vertices
from dataforge.datasets.mamma_source import VERTICES, SmplxModel
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




def floor_mask(person: int, frame: int) -> np.ndarray:
    """The vertices of a person on the floor at a frame (every view ships the same)."""
    return np.arange(VERTICES) % 97 == frame + 3 * person


def partner_distance(person: int) -> np.ndarray:
    """Each vertex's distance to the other person at frame 0, the only frame that ships one; some inside the other (negative)."""
    distance: np.ndarray = np.linspace(-0.02, 0.3, VERTICES, dtype=np.float32)
    return distance if person == 0 else distance[::-1].copy()


def seen_mask(camera: str, person: int, frame: int) -> np.ndarray:
    """The vertices a camera sees of a person at a frame."""
    return (np.arange(VERTICES) + 7 * frame + 3 * person + list(CAMERAS).index(camera)) % 4 != 0


def visibility_rate(camera: str, person: int, frame: int) -> float:
    return 0.5 + 0.1 * person + 0.01 * frame + 0.001 * list(CAMERAS).index(camera)


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
                    "vertices2d": np.zeros((VERTICES, 2)),
                    "motion_info": np.str_("unknown"),
                    "sub": 0,
                    "person_idx": person,
                    "vertex_visibility": seen_mask(name, person, frame).astype(np.int32)[:, None],
                    "person_visitiblity_rate": np.float64(visibility_rate(name, person, frame)),
                    "sdf_vertices": partner_distance(person)[:, None] if frame == 0 else [],
                    "floor_contact_mask": floor_mask(person, frame),
                    "camera_normals": np.zeros((VERTICES, 3)),
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
    for model in BODY_FILES:
        raw_asset("MAMMA SMPL-X model (dataforge-download mamma)", model.local(MODEL_ROOT))
    return MODEL_ROOT


def test_mammanets_landmarks_are_512_distinct_body_vertices(models: Path) -> None:
    landmarks: np.ndarray = landmark_vertices(models)
    assert landmarks.shape == (512,) and len(set(landmarks.tolist())) == 512 and landmarks.max() < 10475
    assert landmarks[:5].tolist() == [0, 9383, 5847, 5060, 3500]  # the release's matrix, row by row


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
    dataset.prefetch(identity, sequence, force=False)  # the runner copies and indexes the next sequence's shards while one converts
    assert (tmp_path / "index" / f"{identity.sequence_key}.json").is_file()
    dataset.convert(identity, sequence, force=False)
    assert not (dataset.stage_root / identity.recording_id).exists()  # the local copy goes once the cameras are read
    targets: dict[str, Path] = {layer: target for layer, target in dataset.targets(identity).items() if target.is_file()}
    assert sorted(targets) == sorted(["base", "body_pose", "body_mesh", "projections", "masks", "contacts", "visibility"])
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


def test_a_render_sequence_carries_its_peoples_contact_and_what_each_camera_sees(
    tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path
) -> None:
    del nvenc_ffmpeg
    write_sequence(tmp_path / "raw")
    dataset: MammaSynDataset = syn_dataset(tmp_path / "raw", tmp_path, monkeypatch, models)
    ((identity, sequence),) = dataset.discover()
    dataset.convert(identity, sequence, force=False)
    targets: dict[str, Path] = dataset.targets(identity)
    times: list[int] = frame_times(np.array(FRAMES, dtype=np.int64), 6).tolist()
    mesh = read_back(targets["body_mesh"])

    contacts = read_back(targets["contacts"])
    for person in (0, 1):
        path: str = schema.person_floor_contact_path(person)
        rows = column_rows(contacts, f"{path}:dataforge.VertexContact:in_contact")
        assert rows[schema.TIMELINE].cast("int64").to_pylist() == times  # every frame; an empty list where the person is absent
        assert rows.column(1).to_pylist() == [[] if (person, frame) == ABSENT else floor_mask(person, frame).tolist() for frame in FRAMES]
        points: list[list[list[float]]] = column_rows(contacts, f"{path}:Points3D:positions").column(1).to_pylist()
        vertices: list[list[list[float]]] = column_rows(mesh, f"{schema.person_body_path(person, 'mesh')}:Mesh3D:vertex_positions").column(1).to_pylist()
        for frame, drawn, posed in zip(FRAMES, points, vertices, strict=True):
            expected: np.ndarray = np.zeros((0, 3)) if (person, frame) == ABSENT else np.array(posed)[floor_mask(person, frame)]
            np.testing.assert_allclose(np.array(drawn).reshape(-1, 3), expected, atol=1e-6)  # on the posed body
        assert static_values(contacts, path)["n_vertices"] == VERTICES

    partner: str = schema.person_contact_path(0, 1)
    distances = column_rows(contacts, f"{partner}:dataforge.VertexContact:signed_distance").column(1).to_pylist()
    np.testing.assert_array_equal(np.array(distances[0], dtype=np.float16), partner_distance(0).astype(np.float16))
    assert distances[1:] == [[], []]  # only frame 0 ships a distance
    flags = column_rows(contacts, f"{partner}:dataforge.VertexContact:in_contact").column(1).to_pylist()
    assert flags[0] == (partner_distance(0) < 0.01).tolist()  # inside the other person counts as contact
    assert static_values(contacts, partner)["contact_threshold_m"] == 0.01
    assert len(column_rows(contacts, f"{partner}:Points3D:positions").column(1).to_pylist()[0]) == int((partner_distance(0) < 0.01).sum())
    assert len(column_rows(contacts, f"{schema.person_contact_path(1, 0)}:dataforge.VertexContact:signed_distance")) == len(FRAMES)
    assert recording_properties(contacts, "contacts")["contact_threshold_m"] == 0.01

    visibility = read_back(targets["visibility"])
    landmarks: np.ndarray = landmark_vertices(models)
    for rig, name in enumerate(CAMERAS, start=1):
        cam_int, cam_ext = camera_matrices(name)
        for person in (0, 1):
            path = schema.person_visibility_path(rig, 0, person)
            visible = column_rows(visibility, f"{path}:dataforge.VertexVisibility:visible").column(1).to_pylist()
            assert visible == [[] if (person, frame) == ABSENT else seen_mask(name, person, frame).tolist() for frame in FRAMES]
            rates = column_rows(visibility, f"{path}:dataforge.VertexVisibility:visibility_rate").column(1).to_pylist()
            np.testing.assert_allclose([row[0] for row in rates if row], [visibility_rate(name, person, f) for f in FRAMES if (person, f) != ABSENT], rtol=1e-6)
            drawn = column_rows(visibility, f"{path}:Points2D:positions").column(1).to_pylist()
            colours = column_rows(visibility, f"{path}:Points2D:colors").column(1).to_pylist()
            vertices = column_rows(mesh, f"{schema.person_body_path(person, 'mesh')}:Mesh3D:vertex_positions").column(1).to_pylist()
            for frame, uv, colour, posed in zip(FRAMES, drawn, colours, vertices, strict=True):
                if (person, frame) == ABSENT:
                    assert uv == []
                    continue
                in_camera: np.ndarray = np.array(posed)[landmarks] @ cam_ext[:3, :3].T + cam_ext[:3, 3]
                pixels: np.ndarray = in_camera @ cam_int.T
                np.testing.assert_allclose(np.array(uv), pixels[:, :2] / pixels[:, 2:], atol=1e-3)  # the landmarks through the camera
                assert len(set(colour)) == 2 and colour.count(colour[0]) in (seen_mask(name, person, frame)[landmarks].sum(), 512 - seen_mask(name, person, frame)[landmarks].sum())
            statics = visibility.reader(index=None, contents=path).to_arrow_table()
            assert statics[f"{path}:landmark_vertex_ids"].to_pylist()[0] == landmarks.tolist()


def test_a_sequence_whose_other_layers_exist_writes_the_label_layers_from_its_shards_in_place(
    tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path
) -> None:
    # The labels-only pass the corpus gets: no staged copy, the label pickles and one PNG header per camera read in place.
    del nvenc_ffmpeg
    write_sequence(tmp_path / "raw")
    dataset: MammaSynDataset = syn_dataset(tmp_path / "raw", tmp_path, monkeypatch, models)
    ((identity, sequence),) = dataset.discover()
    dataset.convert(identity, sequence, force=False)
    targets: dict[str, Path] = dataset.targets(identity)
    full: dict[str, pa.Table] = {}
    for layer in ("contacts", "visibility"):
        full[layer] = read_back(targets[layer]).reader(index=schema.TIMELINE).to_arrow_table()
        targets[layer].unlink()
    dataset.prefetch(identity, sequence, force=False)
    assert not (dataset.stage_root / identity.recording_id).exists()  # nothing staged for labels alone
    dataset.convert(identity, sequence, force=False)
    for layer, table in full.items():
        assert read_back(targets[layer]).reader(index=schema.TIMELINE).to_arrow_table().equals(table)


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
BEDLAM_MOTIONS: tuple[list[str], list[str]] = (
    ["kneel", "1", "/ps/project/datasets/AMASS/amass_march_2022/mosh_results/smplx_neutral/KIT/3/kneel_stageii.npz", "100.0", "0", "591"],
    ["kick", "29", "/ps/project/datasets/AMASS/amass_march_2022/mosh_results/smplx_neutral/CMU/kick_stageii.npz", "120.0", "0", "526"],
)
"""Each person's motion_info as BEDLAM ships it."""


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


def bedlam_camera(frame: int, person: int, size: tuple[int, int] = BEDLAM_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """The zoom: K changes every frame, its principal point at the centre of the camera's ``size`` image; each person's world
    sits at its own place in front of the camera."""
    focal: float = 300.0 + 10.0 * frame
    cam_ext: np.ndarray = np.eye(4)
    cam_ext[:3, 3] = [0.8 * person - 0.4, 0.0, 5.0]
    return np.array([[focal, 0.0, size[0] / 2], [0.0, focal, size[1] / 2], [0.0, 0.0, 1.0]]), cam_ext


def write_bedlam_batch(root: Path, batch: str = BATCH, *, portrait: bool = False) -> Path:
    """A batch as BEDLAM ships it. ``portrait``: the camera's image is BEDLAM_SIZE turned upright (K for 240x320), and the
    shipped PNGs and masks are that image turned a quarter counter-clockwise, as in BEDLAM's closeup batches."""
    directory: Path = root / "MammaSyn" / "MammaSyn-Singles" / "BEDLAM_MASKS_WD" / batch
    directory.mkdir(parents=True)
    members: list[tuple[str, bytes]] = []
    width, height = BEDLAM_SIZE[::-1] if portrait else BEDLAM_SIZE
    for clip, frames in CLIPS.items():
        for frame in frames:
            people: dict[int, dict[str, object]] = {}
            mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
            for person in (0, 1):
                mask[: height // 2, person * width // 2 : (person + 1) * width // 2] = person + 1  # the mask shows both, always
                if (clip, frame, person) == ("seq_000000", 10, 1):
                    continue  # BEDLAM's labels leave out a person its mask still has
                pose, betas, trans, _, joints = bedlam_forward(frame, person)
                cam_int, cam_ext = bedlam_camera(frame, person, (width, height))
                people[person] = {
                    "pose_world": pose,
                    "shape": betas,
                    "trans_world": trans,
                    "cam_int": cam_int,
                    "cam_ext": cam_ext,
                    "gender": "female" if person == 0 else "male",
                    "joints3d": joints @ cam_ext[:3, :3].T + cam_ext[:3, 3],
                    "person_idx": f"{person:02d}",
                    "motion_info": np.array(BEDLAM_MOTIONS[person]),
                    "floor_contact_mask": floor_mask(person, frame),
                    "vertex_visibility": seen_mask("IOI_01", person, frame).astype(np.float64)[:, None],
                    "person_visitiblity_rate": np.float64(visibility_rate("IOI_01", person, frame)),
                    "camera_normals": np.zeros((VERTICES, 3)),
                }
            key: str = f"/ps/project/bedlam/public/b0_all/{batch}/png/{clip}/{clip}_{frame:04d}"
            rgba: np.ndarray = np.full((height, width, 4), 255, dtype=np.uint8)
            if portrait:
                rgba, mask = np.rot90(rgba), np.rot90(mask)  # shipped turned a quarter counter-clockwise
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
    dataset.prefetch(identity, clip, force=False)  # the runner reads the next clip's members while one converts
    for shard in clip.directory.glob("*.tar"):
        shard.rename(shard.with_suffix(".moved"))  # the convert reads the clip from the prefetch alone
    dataset.convert(identity, clip, force=False)
    targets: dict[str, Path] = {layer: target for layer, target in dataset.targets(identity).items() if target.is_file()}
    assert sorted(targets) == sorted(["base", "body_pose", "body_mesh", "projections", "masks", "contacts", "visibility"])
    frames: list[int] = [0, 5, 10]
    times: list[int] = frame_times(np.array(frames, dtype=np.int64), 30).tolist()

    base = read_back(targets["base"])
    assert recording_properties(base, "capture")["fps"] == 6  # the stream's rate; the 30 fps numbering is in clock_source
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


def test_a_bedlam_clip_carries_its_peoples_floor_contact_what_the_camera_sees_and_their_motion(
    tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path
) -> None:
    del nvenc_ffmpeg
    write_bedlam_batch(tmp_path / "raw")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family="syn-bedlam", root=tmp_path / "raw", model_root=models, index_root=tmp_path / "index").setup()
    identity, clip = dataset.discover()[0]
    dataset.convert(identity, clip, force=False)
    targets: dict[str, Path] = dataset.targets(identity)
    frames: list[int] = [0, 5, 10]
    absent: tuple[int, int] = (1, 10)  # the labels leave person 1 out at frame 10

    contacts = read_back(targets["contacts"])
    for person in (0, 1):
        flags = column_rows(contacts, f"{schema.person_floor_contact_path(person)}:dataforge.VertexContact:in_contact").column(1).to_pylist()
        assert flags == [[] if (person, frame) == absent else floor_mask(person, frame).tolist() for frame in frames]
    assert not any(":dataforge.VertexContact:signed_distance" in name for name in contacts.reader(index=schema.TIMELINE).to_arrow_table().column_names)

    visibility = read_back(targets["visibility"])
    mesh = read_back(targets["body_mesh"])
    landmarks: np.ndarray = landmark_vertices(models)
    for person in (0, 1):
        path: str = schema.person_visibility_path(1, 0, person)
        visible = column_rows(visibility, f"{path}:dataforge.VertexVisibility:visible").column(1).to_pylist()
        assert visible == [[] if (person, frame) == absent else seen_mask("IOI_01", person, frame).tolist() for frame in frames]
        drawn = column_rows(visibility, f"{path}:Points2D:positions").column(1).to_pylist()
        vertices = column_rows(mesh, f"{schema.person_body_path(person, 'mesh')}:Mesh3D:vertex_positions").column(1).to_pylist()
        cam_int, _ = bedlam_camera(5, person)
        pixels: np.ndarray = np.array(vertices[1])[landmarks] @ cam_int.T  # the camera-frame body through frame 5's K
        np.testing.assert_allclose(np.array(drawn[1]), pixels[:, :2] / pixels[:, 2:], atol=1e-3)

    pose = read_back(targets["body_pose"])
    for person in (0, 1):
        statics = pose.reader(index=None, contents=schema.person_path(person)).to_arrow_table()
        motion: list[str] = BEDLAM_MOTIONS[person]
        assert statics[f"{schema.person_path(person)}:motion_info"].to_pylist()[0] == motion
        assert (static_values(pose, schema.person_path(person))["motion_action"], static_values(pose, schema.person_path(person))["motion_source"]) == (motion[0], motion[2])


def test_a_clip_whose_other_layers_exist_writes_the_label_layers_from_its_labels_alone(
    tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path
) -> None:
    # The labels-only pass the corpus gets: base and masks exist, so the clip reads only its label pickles and one PNG header.
    del nvenc_ffmpeg
    write_bedlam_batch(tmp_path / "raw")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family="syn-bedlam", root=tmp_path / "raw", model_root=models, index_root=tmp_path / "index").setup()
    identity, clip = dataset.discover()[0]
    dataset.convert(identity, clip, force=False)
    targets: dict[str, Path] = dataset.targets(identity)
    full: dict[str, pa.Table] = {}
    for layer in ("contacts", "visibility"):
        full[layer] = read_back(targets[layer]).reader(index=schema.TIMELINE).to_arrow_table()
        targets[layer].unlink()
    dataset.prefetch(identity, clip, force=False)
    for shard in clip.directory.glob("*.tar"):
        shard.rename(shard.with_suffix(".moved"))  # the convert reads the clip from the prefetch alone
    dataset.convert(identity, clip, force=False)
    for layer, table in full.items():
        assert read_back(targets[layer]).reader(index=schema.TIMELINE).to_arrow_table().equals(table)


CLOSEUP_BATCH: str = "20221011_1_250_batch01hand_closeup_suburb_a"


def test_a_bedlam_closeup_clip_is_turned_upright_to_match_its_intrinsics(
    tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path
) -> None:
    """BEDLAM renders its closeup batches in portrait and ships the PNGs turned to landscape; K is for the portrait image."""
    del nvenc_ffmpeg
    write_bedlam_batch(tmp_path / "raw", CLOSEUP_BATCH, portrait=True)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family="syn-bedlam", root=tmp_path / "raw", model_root=models, index_root=tmp_path / "index").setup()
    identity, clip = dataset.discover()[0]
    dataset.convert(identity, clip, force=False)
    targets: dict[str, Path] = dataset.targets(identity)

    base = read_back(targets["base"])
    resolutions = column_rows(base, f"{schema.pinhole_path(1, 0)}:Pinhole:resolution").column(1).to_pylist()
    assert [list(row[0]) for row in resolutions] == [[240.0, 320.0]] * 3  # portrait, as K's principal point (120, 160) says

    masks = read_back(targets["masks"])
    blob = column_rows(masks, f"{schema.person_mask_path(1, 0, 0)}:EncodedImage:blob").column(1).to_pylist()[0][0]
    quarter: np.ndarray = np.asarray(Image.open(io.BytesIO(bytes(blob))))
    assert quarter.shape == (80, 60)  # a quarter of 240x320, upright
    rows, cols = np.nonzero(quarter)
    assert rows.max() < 40 and cols.max() < 30  # person 0 back in the top-left quarter of the upright image


def test_a_bedlam_clip_whose_intrinsics_do_not_centre_on_its_image_fails(
    tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, models: Path
) -> None:
    """A portrait batch not named closeup would keep its landscape PNGs under portrait intrinsics: fail it, do not publish it."""
    del nvenc_ffmpeg
    write_bedlam_batch(tmp_path / "raw", portrait=True)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family="syn-bedlam", root=tmp_path / "raw", model_root=models, index_root=tmp_path / "index").setup()
    identity, clip = dataset.discover()[0]
    with pytest.raises(ValueError, match=r"principal point \(120\.0, 160\.0\).*centre of the 320x240 image"):
        dataset.convert(identity, clip, force=False)


def test_the_syn_scenes_keep_the_visibility_landmarks_in_the_camera_panes() -> None:
    # The landmarks are pixels of one camera's image: in 3D they would sit on that camera's frustum image.
    for family, rigs in (("syn", 8), ("syn-bedlam", 1)):
        dataset = MammaConfig(family=family).setup()
        for blueprint in (dataset.default_blueprint(), dataset.table_blueprint()):
            contents: list[str] = list(scene(blueprint).contents)  # type: ignore[arg-type]
            assert f"- {schema.pinhole_path(rigs, 0)}/visibility/**" in contents


def test_bedlam_puts_the_camera_beside_the_scene() -> None:
    # Closeup clips are portrait: a pane in the strip under the scene showed them at ~102x180 px in a 1920x1080 viewer.
    root = MammaConfig(family="syn-bedlam").setup().default_blueprint().root_container
    assert isinstance(root, rrb.Horizontal)
    assert [type(view) for view in root.contents] == [rrb.Spatial3DView, rrb.Spatial2DView]  # type: ignore[union-attr]
