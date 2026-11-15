"""Ego-Exo4D end to end on a synthetic take laid out as the egoexo CLI writes it (real HM fit, real Aria calibration)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURES, raw_asset, read_chunks, vrs_file
from test_egoexo4d_download import entry, release

from dataforge import aria, paths, schema
from dataforge.datasets.egoexo4d import Egoexo4dConfig, Egoexo4dDataset
from dataforge.datasets.egoexo4d_body import HM_FILE, SMPLH_FILE, SMPLX_FILE
from dataforge.datasets.egoexo4d_layers import EGO_RIG, SIDECAR, CamerasSidecar, FrameSidecar, read_sidecar, restored_size, write_sidecar
from dataforge.datasets.egoexo4d_source import Take
from dataforge.meshes import BODY_MESH_STRIDE

TAKE: str = "cmu_bike02_4"
FRAMES: int = 12
MODEL_ROOT: Path = Path(os.environ.get("DATAFORGE_EGOEXO4D_MODEL_ROOT", str(paths.raw_root() / "egoexo4d")))
GOPRO_HEADER: str = (
    "cam_uid,graph_uid,tx_world_cam,ty_world_cam,tz_world_cam,qx_world_cam,qy_world_cam,qz_world_cam,qw_world_cam,image_width,image_height,"
    "intrinsics_type,intrinsics_0,intrinsics_1,intrinsics_2,intrinsics_3,intrinsics_4,intrinsics_5,intrinsics_6,intrinsics_7,"
    "start_frame_idx,end_frame_idx,quality"
)


def write_video(ffmpeg: Path, path: Path, size: str, *, gray: bool, padded: bool = False) -> None:
    """A 12-frame H.264 clip with B-frames, as the release encodes its frame-aligned videos.

    padded blacks out two frames of every three, the image in the middle one, as the release pads the 10 Hz eye cameras,
    plus frame 4, a dropped sample: images in frames 1, 7 and 10.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    padding: str = ",format=yuv420p,lutyuv=y=0:u=128:v=128:enable='not(eq(mod(n,3),1))+eq(n,4)'" if padded else ""
    source: str = f"testsrc2=size={size}:rate=30" + padding + (",format=gray" if gray else "")
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-frames:v",
            str(FRAMES),
            "-c:v",
            "libx264",
            "-bf",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def synthetic_take(root: Path, ffmpeg: Path, *, slam_size: str = "480x640") -> list[str]:
    """Lay out one take; return the take files a real manifest would list for it.

    slam_size 640x480 ships the SLAM videos stretched to the readout's shape, as about one real take in seven does.
    """
    take_dir: Path = root / "takes" / TAKE
    videos: dict[str, tuple[str, bool]] = {
        "cam01.mp4": ("3840x2160", False),
        "cam02.mp4": ("3840x2160", False),
        "aria01_214-1.mp4": ("1408x1408", False),
        "aria01_1201-1.mp4": (slam_size, True),
        "aria01_1201-2.mp4": (slam_size, True),
        "aria01_211-1.mp4": ("640x240", True),
    }
    for name, (size, gray) in videos.items():
        write_video(ffmpeg, take_dir / "frame_aligned_videos" / name, size, gray=gray, padded=name == "aria01_211-1.mp4")
    entry = {
        "take_name": TAKE,
        "take_uid": "take-uid",
        "root_dir": f"takes/{TAKE}",
        "capture_uid": "capture-uid",
        "timesync_start_idx": 3,
        "timesync_end_idx": 3 + FRAMES,
        "task_name": "Fix a flat",
        "parent_task_name": "Bike Repair",
        "university_name": "cmu",
        "capture": {"capture_name": "cmu_bike02", "cameras": [{"cam_id": "cam01", "is_ego": False}, {"cam_id": "aria01", "is_ego": True}]},
        "frame_aligned_videos": {
            "cam01": {"0": {"relative_path": "frame_aligned_videos/cam01.mp4", "readable_stream_id": "0"}},
            "cam02": {"0": {"relative_path": "frame_aligned_videos/cam02.mp4", "readable_stream_id": "0"}},
            "aria01": {
                stream: {"relative_path": f"frame_aligned_videos/aria01_{stream_id}.mp4", "readable_stream_id": stream}
                for stream, stream_id in (("rgb", "214-1"), ("slam-left", "1201-1"), ("slam-right", "1201-2"), ("et", "211-1"))
            },
        },
    }
    (root / "takes.json").write_text(json.dumps([entry]))
    times: np.ndarray = 5_000_000_000 + np.arange(FRAMES + 6, dtype=np.int64) * 33_333_333
    capture: Path = root / "captures/cmu_bike02/timesync.csv"
    capture.parent.mkdir(parents=True)
    capture.write_text("cam01_pts,aria01_214-1_capture_timestamp_ns\n" + "".join(f"{i},{stamp}\n" for i, stamp in enumerate(times)))
    trajectory: Path = take_dir / "trajectory"
    trajectory.mkdir(parents=True)
    # GoPros on a 3 m ring around the origin looking at it; the Aria walks along x at 1.6 m.
    rows: list[str] = []
    for index, angle in enumerate((0.0, np.pi / 2)):
        from scipy.spatial.transform import Rotation

        position = np.array([3 * np.cos(angle), 3 * np.sin(angle), 1.2])
        forward = -position / np.linalg.norm(position)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        quaternion = Rotation.from_matrix(np.column_stack([right, down, forward])).as_quat()
        rows.append(
            f"cam{index + 1:02d},g,{position[0]},{position[1]},{position[2]},{quaternion[0]},{quaternion[1]},{quaternion[2]},{quaternion[3]},3840,2160,"
            "KANNALABRANDTK3,1761.27,1761.27,1920,1080,0.0369,0.0571,-0.0571,0.0184,-1,-1,1"
        )
    rows.append("cam03,g,0,0,0,0,0,0,1,3840,2160,KANNALABRANDTK3,1700,1700,1920,1080,0.03,0.05,-0.05,0.01,-1,-1,0")
    (trajectory / "gopro_calibs.csv").write_text("\n".join([GOPRO_HEADER, *rows]) + "\n")
    stamps_us: np.ndarray = np.arange(int(times[0] // 1000) - 2000, int(times[-1] // 1000) + 2000, 1000)
    lines: list[str] = [
        "graph_uid,tracking_timestamp_us,utc_timestamp_ns,tx_world_device,ty_world_device,tz_world_device,qx_world_device,qy_world_device,qz_world_device,qw_world_device,quality_score"
    ]
    lines += [f"g,{stamp},-1,{(stamp - stamps_us[0]) * 1e-6},0.0,1.6,0,0,0,1,1.0" for stamp in stamps_us]
    (trajectory / "closed_loop_trajectory.csv").write_text("\n".join(lines) + "\n")
    calib_json: str = (FIXTURES / "aria/gen1-hot3d-P0015_179e1b84-calib.json").read_text()
    (take_dir / "aria01_noimagestreams.vrs").write_bytes(vrs_file({214: {}}, [], file_tags={"calib_json": calib_json}))
    fit: Path = root / "hm" / TAKE / HM_FILE
    fit.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "egoexo4d/cmu_bike02_4-first12.npz", fit)
    for name in (SMPLH_FILE, SMPLX_FILE):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MODEL_ROOT / name, root / name)
    return [f"takes/{TAKE}/frame_aligned_videos/{name}" for name in videos] + [
        f"takes/{TAKE}/trajectory/gopro_calibs.csv",
        f"takes/{TAKE}/trajectory/closed_loop_trajectory.csv",
        f"takes/{TAKE}/aria01_noimagestreams.vrs",
    ]


def staged_take(
    tmp_path: Path,
    ffmpeg: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    slam_size: str = "480x640",
    keep_raw: bool = False,
    frame_limit: int | None = None,
) -> tuple[Egoexo4dDataset, Path, list[str]]:
    """The synthetic take under ``tmp_path/raw`` and its dataset, converting into ``tmp_path/out``.

    The release's manifests list the take's files at their sizes on disk, so every fetch finds them complete. Returns the
    dataset, the raw root and the take files.
    """
    for name in (SMPLH_FILE, SMPLX_FILE):
        raw_asset("Ego-Exo4D-HM body model (dataforge-download egoexo4d)", MODEL_ROOT / name)
    root: Path = tmp_path / "raw"
    take_files: list[str] = synthetic_take(root, ffmpeg, slam_size=slam_size)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = Egoexo4dConfig(root=root, keep_raw=keep_raw, frame_limit=frame_limit).setup()
    assert isinstance(dataset, Egoexo4dDataset)
    parts: dict[str, list[str]] = {
        "takes": [path for path in take_files if "/frame_aligned_videos/" in path],
        "take_trajectory": [path for path in take_files if "/trajectory/" in path],
        "take_vrs_noimagestream": [path for path in take_files if path.endswith(".vrs")],
    }
    manifests = {part: [entry("take-uid", [(path, (root / path).stat().st_size) for path in files])] for part, files in parts.items()}
    manifests["captures"] = [entry("capture-uid", [("captures/cmu_bike02/timesync.csv", (root / "captures/cmu_bike02/timesync.csv").stat().st_size)])]
    dataset.release = release({}, manifests)
    return dataset, root, take_files


@pytest.mark.integration
@pytest.mark.parametrize("slam_size", ["480x640", "640x480"])
def test_convert_writes_four_layers_and_prunes_the_take(slam_size: str, tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, root, take_files = staged_take(tmp_path, nvenc_ffmpeg, monkeypatch, slam_size=slam_size)
    ((identity, take),) = dataset.discover()
    assert isinstance(take, Take) and identity.recording_id == f"egoexo4d__{TAKE}"
    dataset.convert(identity, take, force=False)

    targets = dataset.targets(identity)
    assert all(target.is_file() for target in targets.values())
    base = read_chunks(targets["base"])
    samples = {str(chunk.entity_path): chunk.num_rows for chunk in base if "VideoStream:sample" in chunk.to_record_batch().schema.names}
    expected = [schema.video_path(rig, 0) for rig in (1, 2)] + [schema.video_path(EGO_RIG, cam) for cam in range(4)]
    assert sorted(samples) == sorted(expected)  # the unlocalized cam03 is left out
    eye: str = schema.video_path(EGO_RIG, 3)
    assert samples.pop(eye) == 3  # the eye cameras' real 10 Hz images (frames 1, 7, 10), not the padding or the dropped sample
    assert set(samples.values()) == {FRAMES}
    projections = {str(chunk.entity_path) for chunk in read_chunks(targets["projections"]) if not chunk.is_static}
    assert projections == {schema.coco133_uv_projected_path(rig, cam) for rig, cam in ((1, 0), (2, 0), (EGO_RIG, 0), (EGO_RIG, 1), (EGO_RIG, 2))}
    mesh = [chunk for chunk in read_chunks(targets["body_mesh"]) if str(chunk.entity_path) == schema.body_path("mesh") and not chunk.is_static]
    assert sum(chunk.num_rows for chunk in mesh) == len(range(0, FRAMES, BODY_MESH_STRIDE))
    sidecars: Path = tmp_path / "out" / "sidecars" / identity.recording_id
    assert (sidecars / SIDECAR).is_file()
    cameras: CamerasSidecar = read_sidecar(sidecars / SIDECAR)[1]
    assert cameras.aria_sizes["camera-slam-left"] == (480, 640)  # a stretched SLAM video is stored upright again
    assert not any((root / path).exists() for path in take_files)  # raw pruned after base
    assert (root / "hm" / TAKE / HM_FILE).is_file()  # the fit stays

    # Derived layers rebuild from the sidecars alone, with the raw take gone.
    for layer in ("body_pose", "body_mesh", "projections"):
        targets[layer].unlink()
    dataset.convert(identity, take, force=False)
    assert all(target.is_file() for target in targets.values())


@pytest.mark.integration
def test_a_preview_that_ends_before_the_first_eye_image_converts(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Frame 0 of the eye video is padding, so a one-frame preview holds no eye image: no fault of the take."""
    dataset, _, _ = staged_take(tmp_path, nvenc_ffmpeg, monkeypatch, keep_raw=True, frame_limit=1)
    ((identity, take),) = dataset.discover()
    dataset.convert(identity, take, force=False)
    base = read_chunks(dataset.targets(identity)["base"])
    samples = {str(chunk.entity_path): chunk.num_rows for chunk in base if "VideoStream:sample" in chunk.to_record_batch().schema.names}
    assert sorted(samples) == sorted([schema.video_path(rig, 0) for rig in (1, 2)] + [schema.video_path(EGO_RIG, cam) for cam in range(3)])
    assert set(samples.values()) == {1}
    assert schema.cam_path(EGO_RIG, 3) in {str(chunk.entity_path) for chunk in base}  # the eye camera keeps its node


@pytest.mark.integration
def test_a_source_longer_than_the_take_fails_before_anything_is_published(
    tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-frames:v would cut the extra frames and the prune would delete the originals: the count check refuses first."""
    dataset, root, take_files = staged_take(tmp_path, nvenc_ffmpeg, monkeypatch)
    takes = json.loads((root / "takes.json").read_text())
    takes[0]["timesync_end_idx"] -= 1  # the timesync rows now cover one frame less than the videos hold
    (root / "takes.json").write_text(json.dumps(takes))
    ((identity, take),) = dataset.discover()
    with pytest.raises(ValueError, match=f"{FRAMES} frames, the take's timesync rows give {FRAMES - 1}"):
        dataset.convert(identity, take, force=False)
    assert not any(target.exists() for target in dataset.targets(identity).values())
    assert not any((tmp_path / "out").rglob(f"*{SIDECAR}*"))
    assert all((root / path).exists() for path in take_files)


def test_restored_size_undoes_the_release_stretch() -> None:
    device = aria.DeviceCalibration.from_json((FIXTURES / "aria/gen1-hot3d-P0015_179e1b84-calib.json").read_text(), "fixture")
    slam = device.camera("camera-slam-left")  # readout 640x480
    assert restored_size(slam, 480, 640) is None  # the usual quarter-turned SLAM video
    assert restored_size(slam, 640, 480) == (480, 640)  # the upright image resized back to the readout's shape
    assert restored_size(device.camera("camera-rgb"), 1408, 1408) is None


def test_the_sidecar_reads_back_what_was_written(tmp_path: Path) -> None:
    sidecar: Path = tmp_path / SIDECAR
    frames = FrameSidecar(np.arange(3, dtype=np.int64), np.tile(np.eye(4), (3, 1, 1)))
    cameras = CamerasSidecar(gopros=[], aria_calib_json="{}", aria_sizes={"camera-rgb": (1408, 1408)})
    write_sidecar(sidecar, cameras, frames)
    read_frames, read_cameras = read_sidecar(sidecar)
    np.testing.assert_array_equal(read_frames.times_ns, frames.times_ns)
    assert read_cameras == cameras


@pytest.mark.integration
def test_a_base_without_its_sidecar_redoes_the_take(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Base and its sidecar are published together; a base found alone (a copy that left the sidecars out) is redone with every layer."""
    dataset, _, _ = staged_take(tmp_path, nvenc_ffmpeg, monkeypatch, keep_raw=True)
    ((identity, take),) = dataset.discover()
    dataset.convert(identity, take, force=False)
    targets = dataset.targets(identity)
    before = {layer: target.stat().st_mtime_ns for layer, target in targets.items()}
    sidecar: Path = tmp_path / "out" / "sidecars" / identity.recording_id / SIDECAR
    sidecar.unlink()
    dataset.convert(identity, take, force=False)
    assert sidecar.is_file()
    assert all(target.stat().st_mtime_ns > before[layer] for layer, target in targets.items())
