"""Async logger: the last-keyframe 2D panel logs its image once per keyframe."""

import inspect
import multiprocessing as mp
import queue
from pathlib import Path

import lietorch_mojo as lietorch
import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float32
from numpy import ndarray
from torch import Tensor

from mast3r_slam import async_logger
from mast3r_slam.async_logger import AsyncRerunLogger, snapshot_current_frame
from mast3r_slam.frame import Frame, SharedKeyframes, SharedStates
from mast3r_slam.log_events import KeyframeSnapshot, LogCurrentFrame, LogEvent, LogMapUpdate
from mast3r_slam.mast3r_utils import frame_to_extrinsics
from mast3r_slam.rerun_log_utils import create_blueprints

H: int = 16
W: int = 16


def _grid_pointmap() -> Float32[Tensor, "hw 3"]:
    """Points of a fronto-parallel plane at z=1 seen by a pinhole with f=W."""
    v_hw: Float32[Tensor, "h w"]
    u_hw: Float32[Tensor, "h w"]
    v_hw, u_hw = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    x_hw: Float32[Tensor, "h w"] = (u_hw - W / 2) / W
    y_hw: Float32[Tensor, "h w"] = (v_hw - H / 2) / W
    return torch.stack([x_hw, y_hw, torch.ones_like(x_hw)], dim=-1).reshape(H * W, 3)


def _make_frame(frame_id: int, feat_dim: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        rgb_tensor=torch.zeros(1, 3, H, W),
        img_shape=torch.tensor([[H, W]], dtype=torch.int),
        img_true_shape=torch.tensor([[H, W]], dtype=torch.int),
        rgb=torch.full((H, W, 3), 0.5),
        world_sim3_cam=lietorch.Sim3.Identity(1, device="cpu"),
        X_canon=_grid_pointmap(),
        C=torch.full((H * W, 1), 10.0),
        feat=torch.zeros(1, 1, feat_dim),
        pos=torch.zeros(1, 1, 2, dtype=torch.long),
    )


def test_snapshot_copies_last_keyframe_rgb_only_when_not_shown() -> None:
    """The keyframe image is copied once; its evolving pointmap on a keyframe change and every N frames."""
    manager = mp.Manager()
    try:
        keyframes: SharedKeyframes = SharedKeyframes(manager, H, W, buffer=4, device="cpu")
        keyframes.append(_make_frame(frame_id=3, feat_dim=keyframes.feat_dim))
        current: Frame = _make_frame(frame_id=4, feat_dim=keyframes.feat_dim)

        unseen: LogCurrentFrame = snapshot_current_frame(current, 4, None, keyframes, shown_last_kf_id=None)
        shown: LogCurrentFrame = snapshot_current_frame(current, 4, None, keyframes, shown_last_kf_id=3)
        refresh: LogCurrentFrame = snapshot_current_frame(current, async_logger.LAST_KEYFRAME_MAPS_EVERY, None, keyframes, shown_last_kf_id=3)
        replaced: LogCurrentFrame = snapshot_current_frame(current, 4, None, keyframes, shown_last_kf_id=1)
    finally:
        manager.shutdown()

    assert unseen.last_kf_frame_id == 3
    assert unseen.last_kf_rgb is not None
    assert unseen.last_kf_X_canon is not None
    assert shown.last_kf_rgb is None
    assert shown.last_kf_X_canon is None and shown.last_kf_C is None, "unchanged keyframe between refreshes"
    assert shown.last_kf_img_shape == (H, W)
    assert refresh.last_kf_rgb is None
    assert refresh.last_kf_X_canon is not None and refresh.last_kf_C is not None
    assert replaced.last_kf_rgb is not None and replaced.last_kf_X_canon is not None


@pytest.mark.parametrize("with_rgb", [False, True])
def test_logger_relogs_pointmap_and_logs_image_only_when_sent(monkeypatch: pytest.MonkeyPatch, with_rgb: bool) -> None:
    """Without a keyframe image the logger still relogs the last keyframe's pointmap."""
    logged_paths: list[str] = []

    def _capture(entity_path: str, *_args: object, **_kwargs: object) -> None:
        logged_paths.append(str(entity_path))

    monkeypatch.setattr(async_logger.rr, "log", _capture)
    monkeypatch.setattr(async_logger.rr, "set_time", lambda *_a, **_k: None)

    rgb: Float32[ndarray, "h w 3"] = np.full((H, W, 3), 0.5, dtype=np.float32)
    X_canon: Float32[ndarray, "hw 3"] = _grid_pointmap().numpy()
    C: Float32[ndarray, "hw 1"] = np.full((H * W, 1), 10.0, dtype=np.float32)
    event: LogCurrentFrame = LogCurrentFrame(
        frame_idx=4,
        timestamp_ns=None,
        world_sim3_cam_data=lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32),
        rgb=rgb,
        X_canon=X_canon,
        C=C,
        img_shape=(H, W),
        last_kf_frame_id=3,
        last_kf_img_shape=(H, W),
        last_kf_rgb=rgb if with_rgb else None,
        last_kf_X_canon=X_canon,
        last_kf_C=C,
    )
    event_queue: queue.Queue[LogEvent] = queue.Queue()
    logger: AsyncRerunLogger = AsyncRerunLogger(event_queue, parent_log_path=Path("world"))
    logger._handle_current_frame(event)

    assert "/last_keyframe/pointmap" in logged_paths
    assert "/last_keyframe/confidence" in logged_paths
    assert ("/last_keyframe/image" in logged_paths) == with_rgb
    assert "world/current_camera/pinhole/image" in logged_paths


def test_blueprint_is_static_and_logger_never_resends_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """One blueprint covers any number of keyframes: no keyframe-count input, no per-keyframe rules, no re-send."""
    assert "n_keyframes" not in inspect.signature(create_blueprints).parameters
    views_3d: list[rrb.Spatial3DView] = []
    origins_2d: list[str] = []
    pending: list[object] = [create_blueprints(Path("world")).root_container]
    while pending:
        item = pending.pop()
        if isinstance(item, rrb.Spatial3DView):
            views_3d.append(item)
        elif isinstance(item, rrb.Spatial2DView):
            origins_2d.append(str(item.origin))
        elif isinstance(item, rrb.Container):
            pending.extend(item.contents)
    # The original video is the one camera image in 2D; the model's crop shows on the frustum in 3D.
    assert origins_2d.count("/video") == 1
    assert "/world/current_camera/pinhole/image" not in origins_2d
    assert sorted(str(view.name) for view in views_3d) == ["Follow", "Overview"]
    for view in views_3d:
        # One static rule: the entity layout keeps 2D-only maps out of /world, so nothing needs excluding.
        assert view.contents == "+ /world/**"

    sent: list[object] = []
    monkeypatch.setattr(async_logger.rr, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(async_logger.rr, "set_time", lambda *_a, **_k: None)
    monkeypatch.setattr(async_logger.rr, "send_blueprint", lambda *a, **_k: sent.append(a))
    identity: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32)
    logger: AsyncRerunLogger = AsyncRerunLogger(queue.Queue(), parent_log_path=Path("world"))
    for kf_idx in range(3):
        keyframe: KeyframeSnapshot = KeyframeSnapshot(
            kf_idx=kf_idx,
            world_sim3_cam_data=identity,
            rgb=np.full((H, W, 3), 0.5, dtype=np.float32),
            X_canon=_grid_pointmap().numpy(),
            C=np.full((H * W, 1), 10.0, dtype=np.float32),
            img_shape=(H, W),
        )
        logger._handle_map_update(LogMapUpdate(frame_idx=kf_idx, timestamp_ns=None, new_keyframes=[keyframe]))
    assert sent == []


def test_entity_layout_each_pinhole_owns_what_projects_in_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under /world only the current camera's image and depth and each keyframe's image are 2D, each under a Pinhole
    that has a value from the run's first time point; every other map lives outside /world."""
    logged: list[tuple[str, str, int]] = []
    frame_now: list[int] = [-1]

    def _set_time(timeline: str, *, sequence: int | None = None, **_kwargs: object) -> None:
        if timeline == "frame" and sequence is not None:
            frame_now[0] = sequence

    monkeypatch.setattr(async_logger.rr, "log", lambda path, entity, *_a, **_k: logged.append((str(path), type(entity).__name__, frame_now[0])))
    monkeypatch.setattr(async_logger.rr, "set_time", _set_time)
    identity: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32)
    rgb: Float32[ndarray, "h w 3"] = np.full((H, W, 3), 0.5, dtype=np.float32)
    X_canon: Float32[ndarray, "hw 3"] = _grid_pointmap().numpy()
    C: Float32[ndarray, "hw 1"] = np.full((H * W, 1), 10.0, dtype=np.float32)
    logger: AsyncRerunLogger = AsyncRerunLogger(queue.Queue(), parent_log_path=Path("world"))
    logger._handle_current_frame(
        LogCurrentFrame(frame_idx=0, timestamp_ns=None, world_sim3_cam_data=identity, rgb=rgb, X_canon=X_canon, C=C, img_shape=(H, W))
    )
    keyframe: KeyframeSnapshot = KeyframeSnapshot(kf_idx=5, world_sim3_cam_data=identity, rgb=rgb, X_canon=X_canon, C=C, img_shape=(H, W))
    logger._handle_map_update(LogMapUpdate(frame_idx=40, timestamp_ns=None, new_keyframes=[keyframe]))

    images_2d: set[str] = {"Image", "EncodedImage", "EncodedDepthImage"}
    under_world_2d: set[str] = {path for path, kind, _ in logged if kind in images_2d and path.startswith("world/")}
    assert under_world_2d == {
        "world/current_camera/pinhole/image",
        "world/current_camera/pinhole/depth",
        "world/keyframes/keyframe-5/pinhole/image",
    }
    # No explicit CoordinateFrame anywhere: in 0.42 one breaks the Follow view's chase camera.
    assert not any(kind == "CoordinateFrame" for _, kind, _ in logged)
    for image_path in under_world_2d:
        pinhole_path: str = image_path.rsplit("/", 1)[0]
        pinhole_frames: list[int] = [frame for path, kind, frame in logged if path == pinhole_path and kind in {"Pinhole", "PinholeWithDistortion"}]
        assert pinhole_frames and min(pinhole_frames) == 0, f"{pinhole_path} needs a Pinhole from the first time point"
    keyframe_image_frames: list[int] = [
        frame for path, kind, frame in logged if path == "world/keyframes/keyframe-5/pinhole/image" and kind in images_2d
    ]
    assert keyframe_image_frames == [40], "the keyframe image is logged at the keyframe's own time"

    logged_paths: set[str] = {path for path, _, _ in logged}
    assert {"/current_maps/pointmap", "/current_maps/confidence"} <= logged_paths
    assert {"/keyframe_maps/keyframe-5/pointmap", "/keyframe_maps/keyframe-5/confidence", "/keyframe_maps/keyframe-5/depth"} <= logged_paths
    assert {"/last_keyframe/image", "/last_keyframe/pointmap", "/last_keyframe/confidence", "/last_keyframe/depth"} <= logged_paths
    # The map is the fused TSDF mesh, not a point cloud per keyframe.
    assert not any(kind == "Points3D" for _, kind, _ in logged)


def test_camera_path_logs_one_polyline_per_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each frame logs its window's polyline so far at /world/path/<k>; a full window starts the next at its last point."""
    strips: list[tuple[str, int]] = []

    def _capture(path: str, entity: object, *_a: object, **_k: object) -> None:
        if isinstance(entity, rr.LineStrips3D):
            assert entity.strips is not None
            strips.append((str(path), len(entity.strips.pa_array.to_pylist()[0])))

    monkeypatch.setattr(async_logger.rr, "log", _capture)
    monkeypatch.setattr(async_logger.rr, "set_time", lambda *_a, **_k: None)
    identity: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32)
    rgb: Float32[ndarray, "h w 3"] = np.full((H, W, 3), 0.5, dtype=np.float32)
    X_canon: Float32[ndarray, "hw 3"] = _grid_pointmap().numpy()
    C: Float32[ndarray, "hw 1"] = np.full((H * W, 1), 10.0, dtype=np.float32)
    logger: AsyncRerunLogger = AsyncRerunLogger(queue.Queue(), parent_log_path=Path("world"))
    monkeypatch.setattr(async_logger, "PATH_WINDOW_FRAMES", 2)
    for frame_idx in range(5):
        logger._handle_current_frame(
            LogCurrentFrame(frame_idx=frame_idx, timestamp_ns=None, world_sim3_cam_data=identity, rgb=rgb, X_canon=X_canon, C=C, img_shape=(H, W))
        )
    # Windows of 2 segments: frames 1-2 grow window 0, frames 3-4 grow window 1 from frame 2's point.
    assert strips == [("world/path/0", 2), ("world/path/0", 3), ("world/path/1", 2), ("world/path/1", 3)]



def test_edges_log_one_two_point_strip_per_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each factor-graph edge is its own segment; a flat list of points would draw one polyline through them all."""
    strips: list[list[int]] = []

    def _capture(_path: str, entity: object, *_a: object, **_k: object) -> None:
        if isinstance(entity, rr.LineStrips3D):
            assert entity.strips is not None
            strips.append([len(strip) for strip in entity.strips.pa_array.to_pylist()])

    monkeypatch.setattr(async_logger.rr, "log", _capture)
    positions_i: Float32[ndarray, "3 3"] = np.arange(9, dtype=np.float32).reshape(3, 3)
    positions_j: Float32[ndarray, "3 3"] = positions_i + 1.0
    logger: AsyncRerunLogger = AsyncRerunLogger(queue.Queue(), parent_log_path=Path("world"))
    logger._log_edges(positions_i, positions_j)
    assert strips == [[2, 2, 2]]


def test_map_update_relogs_only_moved_cameras_and_changed_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracking marks its keyframe dirty on every frame without moving it: an unchanged pose or edge set logs nothing."""
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(async_logger.rr, "log", lambda path, entity, *_a, **_k: logged.append((str(path), type(entity).__name__)))
    monkeypatch.setattr(async_logger.rr, "set_time", lambda *_a, **_k: None)
    identity: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32)
    moved: Float32[ndarray, "8"] = identity.copy()
    moved[0] = 1.0
    rgb: Float32[ndarray, "h w 3"] = np.full((H, W, 3), 0.5, dtype=np.float32)
    X_canon: Float32[ndarray, "hw 3"] = _grid_pointmap().numpy()
    C: Float32[ndarray, "hw 1"] = np.full((H * W, 1), 10.0, dtype=np.float32)
    edges: tuple[Float32[ndarray, "1 3"], Float32[ndarray, "1 3"]] = (np.zeros((1, 3), np.float32), np.ones((1, 3), np.float32))
    logger: AsyncRerunLogger = AsyncRerunLogger(queue.Queue(), parent_log_path=Path("world"))
    keyframe: KeyframeSnapshot = KeyframeSnapshot(kf_idx=5, world_sim3_cam_data=identity, rgb=rgb, X_canon=X_canon, C=C, img_shape=(H, W))
    logger._handle_map_update(LogMapUpdate(frame_idx=1, timestamp_ns=None, new_keyframes=[keyframe], edge_positions=edges))

    def logged_by(update: LogMapUpdate) -> set[str]:
        logged.clear()
        logger._handle_map_update(update)
        return {path for path, _ in logged}

    assert logged_by(LogMapUpdate(frame_idx=2, timestamp_ns=None, pose_updates=[(5, identity.copy())], edge_positions=edges)) == set()
    moved_paths: set[str] = logged_by(LogMapUpdate(frame_idx=3, timestamp_ns=None, pose_updates=[(5, moved)], edge_positions=(edges[0] + 1, edges[1])))
    assert {"world/keyframes/keyframe-5", "world/edges"} <= moved_paths


@pytest.mark.parametrize(("panel_hz", "frames_with_panels"), [(10.0, [0]), (0.0, [0, 1, 2])])
def test_current_frame_panels_follow_the_wall_clock_rate(monkeypatch: pytest.MonkeyPatch, panel_hz: float, frames_with_panels: list[int]) -> None:
    """Back-to-back frames log the current panels at most ``panel_hz`` times a second (0: every frame); the camera
    pose and the path are logged for every frame."""
    logged: list[tuple[str, int]] = []
    frame_now: list[int] = [-1]

    def _set_time(timeline: str, *, sequence: int | None = None, **_kwargs: object) -> None:
        if timeline == "frame" and sequence is not None:
            frame_now[0] = sequence

    monkeypatch.setattr(async_logger.rr, "log", lambda path, *_a, **_k: logged.append((str(path), frame_now[0])))
    monkeypatch.setattr(async_logger.rr, "set_time", _set_time)
    identity: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32)
    rgb: Float32[ndarray, "h w 3"] = np.full((H, W, 3), 0.5, dtype=np.float32)
    X_canon: Float32[ndarray, "hw 3"] = _grid_pointmap().numpy()
    C: Float32[ndarray, "hw 1"] = np.full((H * W, 1), 10.0, dtype=np.float32)
    logger: AsyncRerunLogger = AsyncRerunLogger(queue.Queue(), parent_log_path=Path("world"), panel_hz=panel_hz)
    for frame_idx in range(3):
        logger._handle_current_frame(
            LogCurrentFrame(frame_idx=frame_idx, timestamp_ns=None, world_sim3_cam_data=identity, rgb=rgb, X_canon=X_canon, C=C, img_shape=(H, W))
        )
    assert sorted({frame for path, frame in logged if path == "world/current_camera/pinhole/image"}) == frames_with_panels
    assert sorted({frame for path, frame in logged if path == "/current_maps/confidence"}) == frames_with_panels
    assert sorted({frame for path, frame in logged if path == "world/current_camera"}) == [0, 1, 2]
    assert sorted({frame for path, frame in logged if path == "world/path/0"}) == [1, 2]


def test_logger_crash_is_raised_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception on the logger thread reaches the caller (check() and __exit__), not just stderr."""

    def _fail(_event: LogEvent) -> None:
        raise ValueError("logging failed")

    event_queue: queue.Queue[LogEvent] = queue.Queue()
    logger: AsyncRerunLogger = AsyncRerunLogger(event_queue, parent_log_path=Path("world"))
    monkeypatch.setattr(logger, "_handle_event", _fail)
    with pytest.raises(ValueError, match="logging failed"), logger:
        event_queue.put(LogMapUpdate(frame_idx=0, timestamp_ns=None))
        logger.join(timeout=5.0)

def _random_sim3(n: int, seed: int) -> Float32[Tensor, "n 8"]:
    """Sim3 data ``[t, q, s]`` with unit quaternions and positive scales."""
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    t_n3: Float32[Tensor, "n 3"] = torch.randn(n, 3, generator=generator)
    q_n4: Float32[Tensor, "n 4"] = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator), dim=-1)
    s_n1: Float32[Tensor, "n 1"] = 0.5 + torch.rand(n, 1, generator=generator)
    return torch.cat([t_n3, q_n4, s_n1], dim=-1)


def test_compute_orient_matches_the_per_keyframe_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batched pose read hands auto-orient the poses that converting each keyframe's Frame gives."""
    captured: list[Float32[ndarray, "n 4 4"]] = []

    def fake_auto_orient(world_T_cam: Float32[ndarray, "n 4 4"], **_options: str) -> object:
        captured.append(world_T_cam)
        return type("Orient", (), {"transform": np.eye(4)[:3]})()

    monkeypatch.setattr(async_logger, "auto_orient_and_center_poses", fake_auto_orient)
    n_kf: int = 6
    manager = mp.Manager()
    try:
        keyframes: SharedKeyframes = SharedKeyframes(manager, H, W, buffer=8, device="cpu")
        for i, sim3_8 in enumerate(_random_sim3(n_kf, seed=0)):
            frame: Frame = _make_frame(frame_id=i, feat_dim=keyframes.feat_dim)
            frame.world_sim3_cam = lietorch.Sim3(sim3_8[None])
            keyframes.append(frame)
        expected: Float32[ndarray, "n 4 4"] = np.stack(
            [frame_to_extrinsics(keyframes[i], camera_conventions="RUB").world_T_cam.astype(np.float32) for i in range(n_kf)]
        )
        async_logger.compute_orient(keyframes, n_kf)
    finally:
        manager.shutdown()

    np.testing.assert_array_equal(captured[0], expected)


class _CountingListProxy:
    """A Manager ListProxy stand-in: no ``__iter__``, so iterating it calls ``__getitem__`` (one IPC each) per item."""

    def __init__(self, items: list[int]) -> None:
        self._items: list[int] = items
        self.getitem_calls: int = 0

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int | slice) -> int | list[int]:
        self.getitem_calls += 1
        return self._items[index]


def test_snapshot_edges_reads_each_edge_list_in_one_call() -> None:
    """The edge lists cross the process boundary once per snapshot, not once per edge."""
    manager = mp.Manager()
    try:
        keyframes: SharedKeyframes = SharedKeyframes(manager, H, W, buffer=8, device="cpu")
        sim3_n8: Float32[Tensor, "n 8"] = _random_sim3(4, seed=1)
        for i, sim3_8 in enumerate(sim3_n8):
            frame: Frame = _make_frame(frame_id=i, feat_dim=keyframes.feat_dim)
            frame.world_sim3_cam = lietorch.Sim3(sim3_8[None])
            keyframes.append(frame)
        edges_ii: _CountingListProxy = _CountingListProxy([0, 1, 2, 0])
        edges_jj: _CountingListProxy = _CountingListProxy([1, 2, 3, 3])
        states: SharedStates = SharedStates(manager, H, W, device="cpu")
        states.edges_ii = edges_ii  # pyrefly: ignore[bad-assignment]
        states.edges_jj = edges_jj  # pyrefly: ignore[bad-assignment]
        snapshot = async_logger.snapshot_edges(states, keyframes)
    finally:
        manager.shutdown()

    assert snapshot is not None
    positions_i, positions_j = snapshot
    np.testing.assert_allclose(positions_i, sim3_n8[[0, 1, 2, 0], :3].numpy(), atol=1e-6)
    np.testing.assert_allclose(positions_j, sim3_n8[[1, 2, 3, 3], :3].numpy(), atol=1e-6)
    assert edges_ii.getitem_calls == 1
    assert edges_jj.getitem_calls == 1
