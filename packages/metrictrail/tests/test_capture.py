"""Pure parts of the live capture: DPVO pose conversion, frustums, DINOv3 PCA, the kept pass, and the DPVO observer
(on a CPU DPVO with stand-in ViPE modules, no GPU)."""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import depth_png
import numpy as np
import pytest
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

torch = pytest.importorskip("torch", reason="torch comes from the metrictrail env's cuda feature")

from dpvo.config import DPVOConfig  # noqa: E402
from dpvo.dpvo import DPVO  # noqa: E402
from dpvo.hooks import FrameDepth, FrameEvent  # noqa: E402
from dpvo.net import VONet  # noqa: E402

from metrictrail.capture import (  # noqa: E402
    CapturedPass,
    CaptureObserver,
    CaptureSession,
    DinoPca,
    dpvo_world_T_cam,
    frustum_strips,
    normals_rgb,
    rgb_normals,
)


def test_dpvo_world_T_cam_inverts_dpvo_world_to_camera_poses() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    rotations: Float64[ndarray, "n 3 3"] = Rotation.random(5, random_state=1).as_matrix()
    world_T_cam: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (5, 1, 1))
    world_T_cam[:, :3, :3] = rotations
    world_T_cam[:, :3, 3] = rng.normal(size=(5, 3))
    cam_T_world: Float64[ndarray, "n 4 4"] = np.linalg.inv(world_T_cam)
    dpvo: Float32[ndarray, "n 7"] = np.concatenate([cam_T_world[:, :3, 3], Rotation.from_matrix(cam_T_world[:, :3, :3]).as_quat()], axis=1).astype(
        np.float32
    )
    np.testing.assert_allclose(dpvo_world_T_cam(dpvo), world_T_cam, atol=1e-5)


def test_frustum_strips_one_rectangle_and_four_rays_per_camera() -> None:
    poses: Float32[ndarray, "2 4 4"] = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    poses[1, :3, 3] = [1.0, 0.0, 0.0]
    strips = frustum_strips(poses, np.array([100.0, 100.0, 50.0, 40.0], dtype=np.float32), (80, 100), 0.5)
    assert len(strips) == 10
    assert strips[0].shape == (5, 3)
    np.testing.assert_allclose(strips[0][0], [-0.25, -0.2, 0.5], atol=1e-6)  # top-left corner at depth 0.5
    np.testing.assert_allclose(strips[5][0], [0.75, -0.2, 0.5], atol=1e-6)


def test_dino_pca_is_fixed_after_the_fit() -> None:
    generator = torch.Generator().manual_seed(0)
    first = torch.randn(8, 6, 5, generator=generator)
    pca: DinoPca = DinoPca.fit(first)
    rgb = pca.rgb(first)
    assert rgb.shape == (6, 5, 3) and rgb.dtype == np.uint8
    assert np.array_equal(pca.rgb(first), rgb)
    assert rgb.max() == 255 and rgb.min() == 0


def test_kept_pass_is_the_last_of_the_longest(tmp_path: Path) -> None:
    session = CaptureSession(out_dir=tmp_path, application_id="a", recording_id="r", depth_range_m=(0.3, 3.0))
    assert session.kept_pass() is None
    session.passes = [
        CapturedPass(tmp_path / "pass_0.rrd", 32),
        CapturedPass(tmp_path / "pass_1.rrd", 596),
        CapturedPass(tmp_path / "pass_2.rrd", 596),
    ]
    kept = session.kept_pass()
    assert kept is not None and kept.path is not None and kept.path.name == "pass_2.rrd"


def test_normals_survive_the_8bit_rgb_round_trip() -> None:
    generator = torch.Generator().manual_seed(0)
    normals = torch.nn.functional.normalize(torch.randn(6, 5, 3, generator=generator), dim=-1)
    rgb = normals_rgb(normals)
    assert rgb.dtype == torch.uint8
    back: Float32[ndarray, "6 5 3"] = rgb_normals(rgb.numpy())
    np.testing.assert_allclose(np.linalg.norm(back, axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(back, normals.numpy(), atol=0.01)


def test_kept_normals_follow_the_kept_pass_and_superseded_passes_are_emptied(tmp_path: Path) -> None:
    session = CaptureSession(out_dir=tmp_path, application_id="a", recording_id="r", depth_range_m=(0.3, 3.0))
    assert session.kept_normals() == {}
    image: np.ndarray = np.zeros((2, 2, 3), dtype=np.uint8)
    for marker, frames in enumerate((32, 596, 596)):
        session.pass_frames = frames
        session.open_pass(len(session.passes))
        session.pass_normals[-1][0] = torch.from_numpy(image + marker)
    session.uninstall()
    assert [len(normals) for normals in session.pass_normals] == [0, 0, 1]
    assert session.kept_normals()[0][0, 0, 0] == 2


def test_live_surface_grows_temporal_blocks_then_ends_as_one_mesh(tmp_path: Path) -> None:
    import rerun as rr
    from rerun.chunk import RrdReader

    from metrictrail import entities
    from metrictrail.live_surface import LiveSurface

    rrd: Path = tmp_path / "live.rrd"
    rec = rr.RecordingStream("metrictrail-test", recording_id="live")
    rec.save(rrd)
    surface = LiveSurface(rec, log_every=2)
    session = CaptureSession(
        out_dir=tmp_path,
        application_id="metrictrail-test",
        recording_id="live",
        depth_range_m=(0.3, 3.0),
        live=rec,
        live_surface=surface,
    )
    session.pass_frames = 3
    session.open_pass(0)
    depth: np.ndarray = np.full((48, 64), 2.0, dtype=np.float32)
    rgb: np.ndarray = np.full((48, 64, 3), 128, dtype=np.uint8)
    intrinsics: np.ndarray = np.array([60.0, 60.0, 32.0, 24.0], dtype=np.float32)
    for frame, shift in enumerate((0.0, 0.05, 0.1)):
        rec.set_time(entities.FRAME_TIMELINE, sequence=frame)
        pose: np.ndarray = np.eye(4, dtype=np.float32)
        pose[0, 3] = shift
        surface.integrate(depth, rgb, intrinsics, pose, frame)
    session.uninstall()  # the whole surface as one mesh at frame 2
    rec.flush()
    rec.disconnect()
    assert surface.integrated == 3
    mesh_chunks = [chunk for chunk in RrdReader(rrd).stream() if str(chunk.entity_path).startswith("/" + entities.DPVO_MESH)]
    assert mesh_chunks and not any(chunk.is_static for chunk in mesh_chunks), "on the timeline, so scrubbing shows the growth"
    block_frames = {
        frame
        for chunk in mesh_chunks
        if str(chunk.entity_path).startswith(f"/{entities.DPVO_MESH}/pass_0/blocks/")
        for frame in chunk.to_record_batch().column(entities.FRAME_TIMELINE).to_pylist()
    }
    assert block_frames == {1}, "the blocks after the second integrated frame (log_every=2)"
    full_frames = [
        chunk.to_record_batch().column(entities.FRAME_TIMELINE).to_pylist() for chunk in mesh_chunks if str(chunk.entity_path) == f"/{entities.DPVO_MESH}/pass_0/full"
    ]
    assert full_frames == [[2]], "one mesh at the last integrated frame"


def test_a_recovery_pass_gets_its_own_mesh_subtree_and_hides_the_earlier_one(tmp_path: Path) -> None:
    """Pass 0 tracks 6 frames of a wall 5.5 m away; the recovery pass restarts at frame 0 and tracks 3 frames of a wall
    1 m away (other super-blocks: 64 voxels of 4 cm each). Pass 0's rows stay in the recording but a static clear hides them at every time."""
    import rerun as rr
    from rerun.chunk import RrdReader

    from metrictrail import entities
    from metrictrail.live_surface import LiveSurface

    rrd: Path = tmp_path / "two-pass.rrd"
    rec = rr.RecordingStream("metrictrail-test", recording_id="two-pass")
    rec.save(rrd)
    surface = LiveSurface(rec, log_every=2)
    session = CaptureSession(
        out_dir=tmp_path, application_id="metrictrail-test", recording_id="two-pass", depth_range_m=(0.3, 3.0), live=rec, live_surface=surface
    )
    rgb: np.ndarray = np.full((48, 64, 3), 128, dtype=np.uint8)
    intrinsics: np.ndarray = np.array([60.0, 60.0, 32.0, 24.0], dtype=np.float32)
    for instance, (frames, wall_m) in enumerate(((6, 5.5), (3, 1.0))):
        session.pass_frames = frames
        session.open_pass(instance)
        for frame in range(frames):
            rec.set_time(entities.FRAME_TIMELINE, sequence=frame)
            pose: np.ndarray = np.eye(4, dtype=np.float32)
            pose[0, 3] = 0.05 * frame
            surface.integrate(np.full((48, 64), wall_m, dtype=np.float32), rgb, intrinsics, pose, frame)
    session.uninstall()
    rec.flush()
    rec.disconnect()

    rows: list[tuple[str, bool, list[int]]] = []
    for chunk in RrdReader(rrd).stream():
        path: str = str(chunk.entity_path)
        if path.startswith(f"/{entities.DPVO_MESH}/"):
            batch = chunk.to_record_batch()
            frames_logged = batch.column(entities.FRAME_TIMELINE).to_pylist() if entities.FRAME_TIMELINE in batch.schema.names else []
            rows.append((path, chunk.is_static, frames_logged))
    pass0, pass1 = f"/{entities.DPVO_MESH}/pass_0", f"/{entities.DPVO_MESH}/pass_1"
    blocks0 = {path for path, _, _ in rows if path.startswith(f"{pass0}/blocks/")}
    blocks1 = {path for path, _, _ in rows if path.startswith(f"{pass1}/blocks/")}
    assert blocks0 and blocks1
    assert {path.rsplit("/", 1)[1] for path in blocks0}.isdisjoint(path.rsplit("/", 1)[1] for path in blocks1), "other walls, other blocks"
    assert (pass0, True, []) in rows, "the recovery pass hides pass 0 with a static recursive clear"
    assert not any(static for path, static, _ in rows if path.startswith(pass1)), "the recovery pass stays on the timeline"
    assert [frames for path, _, frames in rows if path == f"{pass1}/full"] == [[2]], "its own last integrated frame"
    assert not any(path == f"{pass0}/full" for path, _, _ in rows), "pass 0 never finished"


class _SlamSystem:
    """Stands in for ``vipe.metrictrail.slam.DPVOSLAMSystem`` (the two methods capture wraps)."""

    def run(self, video_stream: Any, **kwargs: Any) -> None: ...

    def _prepare(self, index: int, *args: Any, **kwargs: Any) -> None: ...


class _FrozenDino:
    """Stands in for ``vipe.metrictrail.dino_fusion.FrozenDinoV3``."""

    def forward(self, dpvo_images: Any) -> None: ...


class _NormalProvider:
    """Stands in for ``metrictrail.moge2.MoGe2NormalProvider``."""

    def predict_metric(self, rgb_bchw: Any, *, focal_px: float) -> None: ...


@pytest.fixture
def stand_in_vipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ViPE modules :meth:`CaptureSession.install` wraps, as stand-ins (the real ones need vipe_ext and the GPU)."""
    modules: dict[str, dict[str, type]] = {
        "vipe": {},
        "vipe.metrictrail": {},
        "vipe.metrictrail.slam": {"DPVOSLAMSystem": _SlamSystem},
        "vipe.metrictrail.dino_fusion": {"FrozenDinoV3": _FrozenDino},
        "metrictrail.moge2": {"MoGe2NormalProvider": _NormalProvider},
    }
    for name, attributes in modules.items():
        module: ModuleType = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["vipe"], "metrictrail", sys.modules["vipe.metrictrail"], raising=False)
    for child in ("slam", "dino_fusion"):
        monkeypatch.setattr(sys.modules["vipe.metrictrail"], child, sys.modules[f"vipe.metrictrail.{child}"], raising=False)


def cpu_dpvo() -> DPVO:
    """A small DPVO on the CPU (random weights; built, never run)."""
    return DPVO(DPVOConfig(buffer_size=24, patches_per_frame=4, gradient_bias=False), VONet(), ht=32, wd=32, device=torch.device("cpu"))


@pytest.mark.usefixtures("stand_in_vipe")
def test_installed_session_observes_every_new_dpvo_and_logs_its_frames(tmp_path: Path) -> None:
    from rerun.chunk import RrdReader

    from metrictrail import entities

    status: list[tuple[int, int, int]] = []
    session = CaptureSession(
        out_dir=tmp_path,
        application_id="metrictrail-test",
        recording_id="observer",
        depth_range_m=(0.3, 3.0),
        on_frame=lambda frame, source, frames, keyframes: status.append((frame, frames, keyframes)),
    )
    session.install()
    assert _SlamSystem.run is not session.restore[0][2]  # the ViPE-level wraps are in place
    session.pass_frames = 5
    tracker: DPVO = cpu_dpvo()
    observer = tracker.observers[-1]
    assert isinstance(observer, CaptureObserver)
    assert [captured.num_frames for captured in session.passes] == [5] and id(tracker) in session.streams

    tracker.n = 1  # as after __call__ added the frame as a keyframe (before the motion probe initialised DPVO)
    event = FrameEvent(
        tstamp=0,
        frame_index=0,
        keyframes_before=0,
        keyframe_added=True,
        image=torch.full((3, 32, 32), 100, dtype=torch.uint8),
        intrinsics=torch.tensor([30.0, 30.0, 16.0, 16.0]),
        depth=FrameDepth(depth=torch.full((32, 32), 2.0), mask=None),
    )
    observer.on_frame(tracker, event)
    observer.on_global_ba(tracker)
    assert status == [(0, 5, 1)]
    assert session.global_ba_calls == 1

    session.uninstall()
    assert session.remove_observer is None and session.restore == []
    assert cpu_dpvo().observers == [] and len(session.passes) == 1  # no longer observed
    paths: set[str] = {str(path) for path in RrdReader(tmp_path / "pass_0.rrd").store().schema().entity_paths()}
    assert {"/" + entities.CAM_RGB, "/" + entities.CAM_DEPTH, "/" + entities.CAM_PATCHES} <= paths
    depth_blobs: list[bytes] = [
        bytes(blob[0])
        for chunk in RrdReader(tmp_path / "pass_0.rrd").store().stream()
        if chunk.entity_path == "/" + entities.CAM_DEPTH
        for blob in chunk.to_record_batch().column("EncodedDepthImage:blob").to_pylist()
    ]
    assert len(depth_blobs) == 1
    np.testing.assert_array_equal(depth_png.decode(depth_blobs[0]), np.full((32, 32), 2000, dtype=np.uint16))  # 2.0 m, in millimetres


@pytest.mark.usefixtures("stand_in_vipe")
def test_a_video_input_puts_the_patches_on_the_source_video_at_its_pts(tmp_path: Path) -> None:
    from rerun.chunk import RrdReader

    from metrictrail import entities
    from metrictrail.rerun_log import CAMERA_PLANE_M, SourceRaster

    raster = SourceRaster(source_hw=(64, 128), resized_hw=(32, 64), crop_left_top=(16, 0))  # 2x down, then a 32x32 centre crop
    session = CaptureSession(
        out_dir=tmp_path,
        application_id="metrictrail-test",
        recording_id="video",
        depth_range_m=(0.3, 3.0),
        video_time_ns=np.arange(10, dtype=np.int64) * 100_000_000,
        source_raster=lambda: raster,
    )
    session.install()
    session.pass_frames = 5
    session.source_frames[0] = 7  # as ViPE's _prepare reports it (a strided clip)
    tracker: DPVO = cpu_dpvo()
    tracker.n = 1
    event = FrameEvent(
        tstamp=0,
        frame_index=0,
        keyframes_before=0,
        keyframe_added=True,
        image=torch.full((3, 32, 32), 100, dtype=torch.uint8),
        intrinsics=torch.tensor([30.0, 30.0, 16.0, 16.0]),
        depth=FrameDepth(depth=torch.full((32, 32), 2.0), mask=None),
    )
    tracker.observers[-1].on_frame(tracker, event)
    session.uninstall()
    store = RrdReader(tmp_path / "pass_0.rrd").store()
    cleared: bool = False
    paths: set[str] = {str(path) for path in store.schema().entity_paths()}
    assert {"/" + entities.INPUT, "/" + entities.INPUT_PATCHES, "/" + entities.CAM_DEPTH} <= paths
    assert "/" + entities.CAM_RGB not in paths and "/" + entities.CAM_PATCHES not in paths
    for chunk in store.stream():
        batch = chunk.to_record_batch()
        if chunk.entity_path == "/" + entities.INPUT and "Pinhole:resolution" in batch.schema.names:
            # The source camera is the moving frustum: the input video plays inside it in the 3D views.
            assert entities.INPUT == entities.DPVO_CURRENT_PINHOLE
            assert batch.column("Pinhole:resolution").to_pylist() == [[[128.0, 64.0]]]
            assert batch.column("Pinhole:image_plane_distance").to_pylist()[0][0] == pytest.approx(CAMERA_PLANE_M)
        if chunk.entity_path == "/" + entities.INPUT_PATCHES and "Points2D:positions" in batch.schema.names:
            assert batch.column("video_time").to_numpy().astype("int64").tolist() == [700_000_000]
            positions: Float32[ndarray, "m 2"] = np.array(batch.column("Points2D:positions").to_pylist()[0], dtype=np.float32)
            assert positions[:, 0].min() >= 32.0 and positions[:, 0].max() <= 96.0  # inside the crop, in source pixels
        if chunk.entity_path == "/" + entities.INPUT_PATCHES and "Clear:is_recursive" in batch.schema.names:
            # Cleared at the next source frame: playback past the tracked progress shows no stale patches.
            assert batch.column("video_time").to_numpy().astype("int64").tolist() == [800_000_000]
            cleared = True
    assert cleared


def test_orient_turns_dpvo_world_upright_with_cameras_looking_level() -> None:
    from metrictrail.capture import dpvo_orient

    # DPVO's world is the first camera's RDF frame: y points down. Cameras walk along x, looking along +z, level.
    world_T_cam: Float32[ndarray, "n 4 4"] = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))
    world_T_cam[:, 0, 3] = np.linspace(0.0, 2.0, 6)
    rotation, translation = dpvo_orient(world_T_cam)
    up_in_world: Float32[ndarray, "3"] = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # camera up = -y (RDF)
    np.testing.assert_allclose(rotation @ up_in_world, [0.0, 0.0, 1.0], atol=1e-5)  # becomes the viewer's +z
    forward: Float32[ndarray, "3"] = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert abs(float(forward[2])) < 1e-5  # looking level, not into the ground
    np.testing.assert_allclose(rotation @ world_T_cam[:, :3, 3].mean(axis=0) + translation, 0.0, atol=1e-5)  # centred
