"""Async logger: the last-keyframe 2D panel logs its image once per keyframe."""

import multiprocessing as mp
import queue
from pathlib import Path

import lietorch_mojo as lietorch
import numpy as np
import pytest
import rerun as rr
import torch
from jaxtyping import Float32
from numpy import ndarray
from torch import Tensor

from mast3r_slam import async_logger
from mast3r_slam.async_logger import AsyncRerunLogger, snapshot_current_frame
from mast3r_slam.config import config
from mast3r_slam.frame import Frame, SharedKeyframes
from mast3r_slam.image_utils import VideoToFrame
from mast3r_slam.log_events import KeyframeSnapshot, LogCurrentFrame, LogEvent
from mast3r_slam.rerun_log_utils import VideoSource

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
    """The keyframe image is copied once; the evolving pointmap is copied every frame."""
    monkeypatch.setitem(config, "use_calib", False)
    manager = mp.Manager()
    try:
        keyframes: SharedKeyframes = SharedKeyframes(manager, H, W, buffer=4, device="cpu")
        keyframes.append(_make_frame(frame_id=3, feat_dim=keyframes.feat_dim))
        current: Frame = _make_frame(frame_id=4, feat_dim=keyframes.feat_dim)

        unseen: LogCurrentFrame = snapshot_current_frame(current, 4, None, keyframes, shown_last_kf_id=None)
        shown: LogCurrentFrame = snapshot_current_frame(current, 4, None, keyframes, shown_last_kf_id=3)
        replaced: LogCurrentFrame = snapshot_current_frame(current, 4, None, keyframes, shown_last_kf_id=1)
    finally:
        manager.shutdown()

    assert unseen.last_kf_frame_id == 3
    assert unseen.last_kf_rgb is not None
    assert shown.last_kf_rgb is None
    assert shown.last_kf_X_canon is not None
    assert shown.last_kf_C is not None
    assert shown.last_kf_img_shape == (H, W)
    assert replaced.last_kf_rgb is not None


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
    logger: AsyncRerunLogger = AsyncRerunLogger(event_queue, parent_log_path=Path("world"), timeline="frame")
    logger._handle_current_frame(event)

    assert "world/last_keyframe/pointmap" in logged_paths
    assert "world/last_keyframe/confidence" in logged_paths
    assert ("world/last_keyframe/image" in logged_paths) == with_rgb
    assert "world/current_camera/pinhole/image" in logged_paths


def test_video_input_logs_frame_references_not_jpegs(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a source video, each camera image is a reference to its own frame, offset from the first sample."""
    logged: list[tuple[str, object]] = []

    def _capture(entity_path: str, entity: object, *_args: object, **_kwargs: object) -> None:
        logged.append((str(entity_path), entity))

    monkeypatch.setattr(async_logger.rr, "log", _capture)
    monkeypatch.setattr(async_logger.rr, "set_time", lambda *_a, **_k: None)
    monkeypatch.setattr(async_logger.rr, "send_blueprint", lambda *_a, **_k: None)
    monkeypatch.setattr("mast3r_slam.rerun_log_utils.rr.log", _capture)

    video: VideoSource = VideoSource(
        entity_path=Path("/video"),
        frame_timestamps_ns=np.array([1_000, 34_000, 67_000, 100_000, 133_000], dtype=np.int64),
        first_sample_ns=1_000,
        video_to_frame=VideoToFrame(scale=(0.25, 0.25), translation=(-2.0, 0.0)),
    )
    identity: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32)
    rgb: Float32[ndarray, "h w 3"] = np.full((H, W, 3), 0.5, dtype=np.float32)
    X_canon: Float32[ndarray, "hw 3"] = _grid_pointmap().numpy()
    C: Float32[ndarray, "hw 1"] = np.full((H * W, 1), 10.0, dtype=np.float32)
    logger: AsyncRerunLogger = AsyncRerunLogger(queue.Queue(), parent_log_path=Path("world"), timeline="video_time", video=video)
    keyframe: KeyframeSnapshot = KeyframeSnapshot(
        kf_idx=0, frame_id=2, world_sim3_cam_data=identity, rgb=rgb, X_canon=X_canon, C=C, img_shape=(H, W)
    )
    logger._log_new_keyframe(keyframe)
    logger._handle_current_frame(
        LogCurrentFrame(
            frame_idx=4, timestamp_ns=133_000, world_sim3_cam_data=identity, rgb=rgb, X_canon=X_canon, C=C, img_shape=(H, W)
        )
    )

    by_path: dict[str, list[object]] = {}
    for path, entity in logged:
        by_path.setdefault(path, []).append(entity)
    image_paths: list[str] = ["world/keyframes/keyframe-0/pinhole/image", "world/last_keyframe/image", "world/current_camera/pinhole/image"]
    # Pointmap and confidence are derived images and stay encoded; camera images are never re-encoded.
    assert not any(isinstance(e, rr.EncodedImage) for path in image_paths for e in by_path[path])

    def _reference_ns(path: str) -> int:
        references = [e for e in by_path[path] if isinstance(e, rr.VideoFrameReference)]
        assert len(references) == 1
        assert references[0].video_reference is not None
        assert references[0].video_reference.pa_array.to_pylist() == ["/video"]
        assert references[0].timestamp is not None
        return int(references[0].timestamp.pa_array.to_pylist()[0])

    assert _reference_ns("world/keyframes/keyframe-0/pinhole/image") == 66_000
    assert _reference_ns("world/last_keyframe/image") == 66_000
    assert _reference_ns("world/current_camera/pinhole/image") == 132_000
    keyframe_transform = next(e for e in by_path["world/keyframes/keyframe-0/pinhole/image"] if isinstance(e, rr.Transform3D))
    assert keyframe_transform.scale is not None
    np.testing.assert_allclose(keyframe_transform.scale.pa_array.to_pylist()[0], [0.25, 0.25, 1.0])
