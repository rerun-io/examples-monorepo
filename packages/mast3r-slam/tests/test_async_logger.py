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
from mast3r_slam.config import config
from mast3r_slam.frame import Frame, SharedKeyframes
from mast3r_slam.log_events import KeyframeSnapshot, LogCurrentFrame, LogEvent, LogMapUpdate
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


def test_snapshot_copies_last_keyframe_rgb_only_when_not_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keyframe image is copied once; its evolving pointmap on a keyframe change and every N frames."""
    monkeypatch.setitem(config, "use_calib", False)
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
    for image_path in under_world_2d:
        pinhole_path: str = image_path.rsplit("/", 1)[0]
        pinhole_frames: list[int] = [frame for path, kind, frame in logged if path == pinhole_path and kind in {"Pinhole", "PinholeWithDistortion"}]
        assert pinhole_frames and min(pinhole_frames) == 0, f"{pinhole_path} needs a Pinhole from the first time point"
    keyframe_image_frames: list[int] = [frame for path, _, frame in logged if path == "world/keyframes/keyframe-5/pinhole/image"]
    assert keyframe_image_frames == [40], "the keyframe image is logged at the keyframe's own time"

    logged_paths: set[str] = {path for path, _, _ in logged}
    assert {"/current_maps/pointmap", "/current_maps/confidence"} <= logged_paths
    assert {"/keyframe_maps/keyframe-5/pointmap", "/keyframe_maps/keyframe-5/confidence", "/keyframe_maps/keyframe-5/depth"} <= logged_paths
    assert {"/last_keyframe/image", "/last_keyframe/pointmap", "/last_keyframe/confidence", "/last_keyframe/depth"} <= logged_paths
    # The map is the fused TSDF mesh, not a point cloud per keyframe.
    assert not any(kind == "Points3D" for _, kind, _ in logged)


def test_camera_path_sends_only_the_open_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each frame logs one strip of at most PATH_CHUNK_SEGMENTS segments; consecutive chunks share an end point."""
    strips: list[tuple[str, int]] = []

    def _capture(path: str, entity: object, *_a: object, **_k: object) -> None:
        if isinstance(entity, rr.LineStrips3D) and str(path).startswith("world/path/"):
            assert entity.strips is not None
            strips.append((str(path), len(entity.strips.pa_array.to_pylist()[0])))

    monkeypatch.setattr(async_logger.rr, "log", _capture)
    monkeypatch.setattr(async_logger.rr, "set_time", lambda *_a, **_k: None)
    identity: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32)
    rgb: Float32[ndarray, "h w 3"] = np.full((H, W, 3), 0.5, dtype=np.float32)
    X_canon: Float32[ndarray, "hw 3"] = _grid_pointmap().numpy()
    C: Float32[ndarray, "hw 1"] = np.full((H * W, 1), 10.0, dtype=np.float32)
    logger: AsyncRerunLogger = AsyncRerunLogger(queue.Queue(), parent_log_path=Path("world"))
    n_frames: int = 2 * async_logger.PATH_CHUNK_SEGMENTS + 5
    for frame_idx in range(n_frames):
        logger._handle_current_frame(
            LogCurrentFrame(frame_idx=frame_idx, timestamp_ns=None, world_sim3_cam_data=identity, rgb=rgb, X_canon=X_canon, C=C, img_shape=(H, W))
        )
    assert len(strips) == n_frames
    assert max(n_points for _, n_points in strips) == async_logger.PATH_CHUNK_SEGMENTS + 1
    assert [path for path, _ in strips[-1:]] == ["world/path/00002"]
    assert strips[async_logger.PATH_CHUNK_SEGMENTS + 1] == ("world/path/00001", 2), "a new chunk starts at the last point"
