"""MAMMA end to end on a synthetic eval sequence laid out as the release ships it (real SMPL-X models, HEVC 4:4:4 videos)."""

import io
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
from conftest import column_rows, raw_asset, read_back, read_chunks, recording_properties, static_values
from PIL import Image

from dataforge import paths, schema
from dataforge.datasets.mamma import MammaConfig, MammaDataset
from dataforge.datasets.mamma_body import BODY_FILES, landmark_vertices
from dataforge.datasets.mamma_source import VERTICES, MammaCamera, MammaSequence, read_camera
from dataforge.meshes import BODY_MESH_STRIDE

pytestmark = pytest.mark.integration

MODEL_ROOT: Path = Path(os.environ.get("DATAFORGE_MAMMA_MODEL_ROOT", str(paths.raw_root() / "mamma-body-models")))
SEQUENCE: str = "250225_Test_Basic_Whip_00001_00002_1"
FRAMES: int = 6
FPS: int = 25
"""Not the release's 30: the converter must take the rate from the sequence, not assume it."""
CAMERAS: dict[str, tuple[int, int]] = {"IOI_01": (320, 240), "IOI_02": (240, 320), "IOI_03": (2400, 1600)}
"""Native sizes: landscape, portrait, and one above the 3.2 MP cap (stored at half size)."""
MARKERS: int = 4
OUT_OF_VIEW: tuple[str, int, int] = ("IOI_02", 1, 2)
"""(camera, person, frame): the one view a body is not in."""


def in_image(camera: str) -> np.ndarray:
    """A camera's ``is_body_in_img``: frames x people."""
    shipped: np.ndarray = np.ones((FRAMES, 2), dtype=bool)
    if camera == OUT_OF_VIEW[0]:
        shipped[OUT_OF_VIEW[2], OUT_OF_VIEW[1]] = False
    return shipped


def vertex_visibility(camera: str) -> np.ndarray:
    """A camera's per-vertex ``visibility`` as the Dance release ships it: frames x people x vertices, float64 0/1."""
    frame, person, vertex = np.meshgrid(np.arange(FRAMES), np.arange(2), np.arange(VERTICES), indexing="ij")
    return ((vertex + 5 * frame + 2 * person + list(CAMERAS).index(camera)) % 3 == 0).astype(np.float64)


def hevc444(ffmpeg: Path, path: Path, width: int, height: int) -> None:
    """A test-pattern clip encoded as the release encodes its videos: HEVC 4:4:4."""
    subprocess.run(
        [str(ffmpeg), "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={FPS}", "-frames:v", str(FRAMES)]
        + ["-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv444p", "-x265-params", "log-level=error", str(path)],
        check=True,
    )


def look_at(position: np.ndarray) -> np.ndarray:
    """World-to-camera (OpenCV axes) of a camera at ``position`` looking at the origin's vertical axis, z up."""
    forward: np.ndarray = np.array([-position[0], -position[1], 0.0]) / np.linalg.norm(position[:2])
    right: np.ndarray = np.cross(forward, [0.0, 0.0, 1.0])
    down: np.ndarray = np.cross(forward, right)
    rotation: np.ndarray = np.stack([right, down, forward])
    cam_T_world: np.ndarray = np.eye(4)
    cam_T_world[:3, :3] = rotation
    cam_T_world[:3, 3] = -rotation @ position
    return cam_T_world.astype(np.float32)


def write_sequence(root: Path, ffmpeg: Path) -> Path:
    """One eval sequence under Extra (the collection with markers): two people, three cameras, masks and markers."""
    directory: Path = root / "mamma_eval_extra" / SEQUENCE
    for sub in ("gt", "videos_crf16", "masks", "markers"):
        (directory / sub).mkdir(parents=True)
    rng: np.random.Generator = np.random.default_rng(1)
    pose: np.ndarray = np.zeros((FRAMES, 2, 165))
    pose[:, :, 0] = np.pi / 2  # stand upright in a z-up world
    trans: np.ndarray = np.zeros((FRAMES, 2, 3))
    trans[:, 0, 0], trans[:, 1, 0] = -0.4, 0.4
    trans[:, :, 2] = 1.0
    trans[:, 0, 1] = np.linspace(0.0, 0.2, FRAMES)
    np.savez(
        directory / "gt/global.npz",
        seq_name=np.array(SEQUENCE),
        frames_len=np.array(FRAMES),
        fps=np.array(FPS),
        people_len=np.array(2),
        subject_name=np.array(["00002", "00001"]),
        gender=np.array(["female", "neutral"]),
        shape=np.zeros((2, 300)),
        pose_world=pose,
        pose_trans_world=trans,
        flat_hand_mean=np.array(True),
        keywords=np.array(["Whip", "Basic"]),
        method_used=np.array("mosh_betas"),
    )
    for index, (name, (width, height)) in enumerate(CAMERAS.items()):
        angle: float = 2 * np.pi * index / len(CAMERAS)
        focal: float = 0.9 * max(width, height)
        np.savez(
            directory / f"gt/{name}.npz",
            cam_name=np.array(name),
            cam_int=np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32),
            cam_ext=look_at(np.array([4 * np.cos(angle), 4 * np.sin(angle), 1.0])),
            cam_img_w=np.array(float(width)),
            cam_img_h=np.array(float(height)),
            cam_portrait=np.array(height > width),
            is_body_in_img=in_image(name),
            visibility=vertex_visibility(name),
            vicon_radial_2=np.array([width / 2, height / 2, 1e-9, 0.0, 0.0]),
        )
        hevc444(ffmpeg, directory / f"videos_crf16/{name}.mp4", width, height)
        with tarfile.open(directory / f"masks/{name}_masks.tar", "w") as tar:
            for frame in range(FRAMES):
                for person in (1, 2):
                    mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
                    mask[: height // 2, (person - 1) * width // 2 : person * width // 2] = 255
                    buffer = io.BytesIO()
                    Image.fromarray(mask).save(buffer, format="PNG")
                    member = tarfile.TarInfo(f"mask_{frame:04d}_{person:02d}.png")
                    member.size = buffer.getbuffer().nbytes
                    buffer.seek(0)
                    tar.addfile(member, buffer)
    np.save(directory / "markers/labels_m37.npy", np.array([f"M{index}" for index in range(MARKERS)]))
    np.save(directory / "markers/vicon_m37.npy", rng.normal(size=(FRAMES, MARKERS, 3)).astype(np.float32))
    np.save(directory / "markers/baseline_m37.npy", rng.normal(size=(FRAMES, MARKERS, 3)).astype(np.float32))
    return directory


@pytest.fixture
def staged(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[MammaDataset, Path]:
    for model in BODY_FILES:
        raw_asset("MAMMA SMPL-X model (dataforge-download mamma)", model.local(MODEL_ROOT))
    root: Path = tmp_path / "raw"
    directory: Path = write_sequence(root, nvenc_ffmpeg)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family="eval", root=root, model_root=MODEL_ROOT).setup()
    assert isinstance(dataset, MammaDataset)
    return dataset, directory


def test_convert_writes_every_layer_on_the_sequence_clock(staged: tuple[MammaDataset, Path]) -> None:
    dataset, directory = staged
    ((identity, sequence),) = dataset.discover()
    assert isinstance(sequence, MammaSequence)
    assert identity.recording_id == f"mamma-eval__extra__{SEQUENCE}"
    dataset.convert(identity, sequence, force=False)
    targets: dict[str, Path] = dataset.targets(identity)
    assert sorted(targets) == sorted(["base", "body_pose", "body_mesh", "projections", "masks", "markers", "visibility"])
    assert all(target.is_file() for target in targets.values())
    expected_times: list[int] = [frame * 1_000_000_000 // FPS for frame in range(FRAMES)]

    base = read_back(targets["base"])
    for rig, (name, (width, height)) in enumerate(CAMERAS.items(), start=1):
        stored: tuple[int, int] = (width // 2, height // 2) if width * height > 3_200_000 else (width, height)
        samples = column_rows(base, f"{schema.video_path(rig, 0)}:VideoStream:sample")
        assert samples[schema.TIMELINE].cast("int64").to_pylist() == expected_times
        values = static_values(base, schema.cam_path(rig, 0))
        assert (values["name"], values["source_width"], values["source_height"]) == (name, width, height)
        assert (values.get("stored_width", width), values.get("stored_height", height)) == stored
    assert recording_properties(base, "capture")["num_frames"] == FRAMES
    episode = base.reader(index=None, contents="/__properties/**").to_arrow_table()
    assert episode["property:episode:keywords"].to_pylist()[0] == ["Whip", "Basic"]

    pose = read_back(targets["body_pose"])
    with np.load(directory / "gt/global.npz") as source:
        for person in range(2):
            rows = column_rows(pose, f"{schema.person_body_path(person, 'smplx')}:pose")
            assert rows[schema.TIMELINE].cast("int64").to_pylist() == expected_times
            np.testing.assert_array_equal(np.array(rows.column(1).to_pylist())[:, 0], source["pose_world"][:, person])
            keypoints = column_rows(pose, f"{schema.person_coco133_xyz_path(person)}:Points3D:positions")
            assert len(keypoints) == FRAMES
            assert static_values(pose, schema.person_path(person))["source"] == "mosh_gt"

    mesh = read_back(targets["body_mesh"])
    for person in range(2):
        rows = column_rows(mesh, f"{schema.person_body_path(person, 'mesh')}:Mesh3D:vertex_positions")
        assert rows[schema.TIMELINE].cast("int64").to_pylist() == expected_times[::BODY_MESH_STRIDE]

    projected = {str(chunk.entity_path) for chunk in read_chunks(targets["projections"]) if not chunk.is_static}
    assert projected == {schema.person_projection_path(rig, 0, person) for rig in (1, 2, 3) for person in range(2)}

    masks = read_back(targets["masks"])
    draw_orders: set[float] = set()
    for rig, (width, height) in enumerate(CAMERAS.values(), start=1):
        stored = (width // 2, height // 2) if width * height > 3_200_000 else (width, height)
        small = (stored[0] // 4, stored[1] // 4)  # a quarter of the video's size, scaled back over it
        for person in range(2):
            path: str = schema.person_mask_path(rig, 0, person)
            rows = column_rows(masks, f"{path}:EncodedImage:blob")
            assert rows[schema.TIMELINE].cast("int64").to_pylist() == expected_times
            image = Image.open(io.BytesIO(bytes(rows.column(1)[0].as_py()[0])))
            assert image.mode == "P" and image.size == small
            ids = np.asarray(image)  # palette indices are the mask: 1 inside, 0 (transparent) outside
            expected = np.zeros(small[::-1], dtype=np.uint8)
            expected[: small[1] // 2, person * small[0] // 2 : (person + 1) * small[0] // 2] = 1
            np.testing.assert_array_equal(ids, expected)
            static = masks.reader(index=None, contents=path).to_arrow_table()
            assert static.column(f"{path}:Transform3D:scale").to_pylist()[0][0] == pytest.approx([stored[0] / small[0], stored[1] / small[1], 1.0])
            # Explicitly translucent: at the default opacity the viewer may draw the transparent background black.
            assert static.column(f"{path}:EncodedImage:opacity").to_pylist()[0][0] == pytest.approx(0.5)
            draw_orders.add(static.column(f"{path}:EncodedImage:draw_order").to_pylist()[0][0])

    assert len(draw_orders) == 2  # one person's mask never covers another's

    markers = read_back(targets["markers"])
    rows = column_rows(markers, f"{schema.markers_path('vicon')}:Points3D:positions")
    np.testing.assert_allclose(np.array(rows.column(1).to_pylist()), np.load(directory / "markers/vicon_m37.npy"))


def write_iphone_scene(root: Path, ffmpeg: Path) -> Path:
    """One markerless iPhone scene: two predicted people, two lens-less cameras (one 4K, stored at 1920x1080)."""
    directory: Path = root / "mamma_markerless_iphones" / "indoors" / "hugging"
    for sub in ("meta", "pred", "videos"):
        (directory / sub).mkdir(parents=True)
    np.savez(
        directory / "meta/global.npz",
        seq_name=np.array("hugging"),
        frames_len=np.array(FRAMES),
        fps=np.array(FPS),
        frame_start=np.array(0),
        frame_end=np.array(FRAMES),
        method_used=np.array("mamma_pred"),
    )
    for person in range(2):
        pose: np.ndarray = np.zeros((FRAMES, 165), dtype=np.float32)
        pose[:, 0] = np.pi / 2
        trans: np.ndarray = np.tile(np.array([0.5 * person, 0.0, 1.0], dtype=np.float32), (FRAMES, 1))
        np.savez(
            directory / f"pred/params_{person:02d}.npz",
            poses=pose,
            trans=trans,
            betas=np.zeros(16, dtype=np.float32),
            gender=np.array("neutral"),
            num_betas=np.array(16),
            flat_hand_mean=np.array(False),
            mocap_frame_rate=np.array(FPS),
        )
    for index, (name, (width, height)) in enumerate({"A001": (3840, 2160), "B001": (640, 360)}.items()):
        np.savez(
            directory / f"meta/{name}.npz",
            img_rel_path=np.array([f"{frame:06d}.jpg" for frame in range(FRAMES)]),
            cam_name=np.array(name),
            cam_ext=look_at(np.array([6.0 * np.cos(index), 6.0 * np.sin(index), 1.0])),
            cam_int=np.array([[0.8 * width, 0, width / 2], [0, 0.8 * width, height / 2], [0, 0, 1]], dtype=np.float32),
            cam_img_w=np.array(float(width)),
            cam_img_h=np.array(float(height)),
            cam_portrait=np.array(False),
            is_body_in_img=np.zeros(0),
            bbox=np.zeros(0),
        )
        hevc444(ffmpeg, directory / f"videos/{name}.mp4", width, height)
    return directory


def test_a_markerless_scene_has_predicted_bodies_and_no_masks(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for model in BODY_FILES:
        raw_asset("MAMMA SMPL-X model (dataforge-download mamma)", model.local(MODEL_ROOT))
    write_iphone_scene(tmp_path / "raw", nvenc_ffmpeg)
    (tmp_path / "raw/mamma_markerless_iphones/outdoors/empty/meta").mkdir(parents=True)  # a scene without labels or videos is skipped
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family="markerless-iphones", root=tmp_path / "raw", model_root=MODEL_ROOT).setup()
    assert isinstance(dataset, MammaDataset)
    ((identity, sequence),) = dataset.discover()
    assert identity.recording_id == "mamma-markerless-iphones__indoors__hugging"
    dataset.convert(identity, sequence, force=False)
    targets: dict[str, Path] = {layer: target for layer, target in dataset.targets(identity).items() if target.is_file()}
    assert sorted(targets) == sorted(["base", "body_pose", "body_mesh", "projections"])  # no masks or markers outside eval
    base = read_back(targets["base"])
    values = static_values(base, schema.cam_path(1, 0))
    assert (values["stored_width"], values["stored_height"], values["camera_model"]) == (1920, 1080, "pinhole")
    pose = read_back(targets["body_pose"])
    # Model output, not ground truth: the markerless bodies live under /world/pred/mamma (schema §3), nothing under /world/gt.
    for layer in ("body_pose", "body_mesh", "projections"):
        assert not [chunk for chunk in read_chunks(targets[layer]) if "/gt/" in str(chunk.entity_path)], layer
    for person in range(2):
        assert schema.person_path(person, pred="mamma") == f"/world/pred/mamma/people/person_{person:02d}"
        assert static_values(pose, schema.person_path(person, pred="mamma"))["source"] == "mamma_pred"
    mesh = read_back(targets["body_mesh"])
    assert len(column_rows(mesh, f"{schema.person_body_path(0, 'mesh', pred='mamma')}:Mesh3D:vertex_positions")) > 0
    projected_path: str = schema.person_projection_path(1, 0, 0, pred="mamma")
    assert projected_path == "/world/rig_01/cam_00/pinhole/pred/mamma/coco133_uv_projected/person_00"
    projected = column_rows(read_back(targets["projections"]), f"{projected_path}:Points2D:positions")
    pixels = np.array(projected.column(1).to_pylist())
    assert pixels.shape == (FRAMES, 133, 2)
    assert np.isfinite(pixels).all() and (pixels.max(axis=(0, 1)) < [1920, 1080]).all()  # the whole body, in stored pixels (not 4K)


def test_an_eval_sequence_missing_a_mask_archive_fails_the_masks_layer(staged: tuple[MammaDataset, Path]) -> None:
    dataset, directory = staged
    ((identity, sequence),) = dataset.discover()
    dataset.convert(identity, sequence, force=False)
    dataset.targets(identity)["masks"].unlink()
    (directory / "masks/IOI_02_masks.tar").unlink()
    with pytest.raises(FileNotFoundError, match="IOI_02_masks.tar"):
        dataset.convert(identity, sequence, force=False)


def test_an_eval_sequence_carries_what_each_camera_sees_of_each_person(staged: tuple[MammaDataset, Path]) -> None:
    dataset, _ = staged
    ((identity, sequence),) = dataset.discover()
    dataset.convert(identity, sequence, force=False)
    targets: dict[str, Path] = dataset.targets(identity)
    visibility = read_back(targets["visibility"])
    mesh = read_back(targets["body_mesh"])
    landmarks: np.ndarray = landmark_vertices(MODEL_ROOT)
    expected_times: list[int] = [frame * 1_000_000_000 // FPS for frame in range(FRAMES)]
    for rig, (name, (width, height)) in enumerate(CAMERAS.items(), start=1):
        camera: MammaCamera = read_camera(sequence.calibration(name))
        scale: float = 0.5 if width * height > 3_200_000 else 1.0
        for person in range(2):
            path: str = schema.person_visibility_path(rig, 0, person)
            rows = column_rows(visibility, f"{path}:dataforge.VertexVisibility:in_image")
            assert rows[schema.TIMELINE].cast("int64").to_pylist() == expected_times
            assert [row[0] for row in rows.column(1).to_pylist()] == in_image(name)[:, person].tolist()
            visible = column_rows(visibility, f"{path}:dataforge.VertexVisibility:visible").column(1).to_pylist()
            assert visible == (vertex_visibility(name)[:, person] == 1).tolist()
            drawn = column_rows(visibility, f"{path}:Points2D:positions").column(1).to_pylist()
            assert [len(row) for row in drawn] == [512] * FRAMES  # every frame, though the mesh is kept at every 3rd
            vertices = column_rows(mesh, f"{schema.person_body_path(person, 'mesh')}:Mesh3D:vertex_positions").column(1).to_pylist()
            for row, posed in zip(range(0, FRAMES, BODY_MESH_STRIDE), vertices, strict=True):
                expected: np.ndarray = camera.project(np.array(posed, dtype=np.float64)[landmarks]) * scale  # through the Vicon lens
                np.testing.assert_allclose(np.array(drawn[row]), expected, atol=1e-3)
