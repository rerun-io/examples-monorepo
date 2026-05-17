from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray

from sapiens2_pose.api.runtime import PoseEstimation
from sapiens2_pose.api.video import SapiensVideoPoseConfig, _make_effective_bbox_pose_fn, run_video_pose_pipeline
from sapiens2_pose.video_gradio_ui import (
    DEFAULT_TENSORRT_ENGINE_ENV_VAR,
    _default_tensorrt_engine_path,
    _format_video_inference_status,
)


def test_video_pipeline_tensorrt_backend_detects_boxes_then_runs_bbox_pose(tmp_path: Path) -> None:
    video_path: Path = tmp_path / "tiny.mp4"
    _write_test_video(video_path, num_frames=2)
    detected_frames: list[tuple[int, int, int]] = []
    posed_bboxes: list[Float32[ndarray, "n 4"]] = []

    def fake_frame_pose_fn(*_args: object, **_kwargs: object) -> PoseEstimation:
        raise AssertionError("TensorRT video mode should route through the supplied bbox pose function.")

    def fake_detect_persons(
        image_rgb: UInt8[ndarray, "h w 3"],
        **_kwargs: object,
    ) -> Float32[ndarray, "n 4"]:
        detected_frames.append((int(image_rgb.shape[0]), int(image_rgb.shape[1]), int(image_rgb.shape[2])))
        return np.asarray([[2.0, 3.0, 20.0, 22.0]], dtype=np.float32)

    def fake_bbox_pose_fn(
        image_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
        **_kwargs: object,
    ) -> PoseEstimation:
        del image_rgb
        bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        posed_bboxes.append(bboxes_f32.copy())
        keypoints: Float32[ndarray, "sapiens_k 2"] = np.zeros((308, 2), dtype=np.float32)
        scores: Float32[ndarray, "sapiens_k"] = np.ones((308,), dtype=np.float32)
        return PoseEstimation(bboxes=bboxes_f32, keypoints=[keypoints], scores=[scores])

    def fake_log_video_fn(
        *,
        video_source: Path | bytes,
        video_log_path: Path,
        timeline: str = "video_time",
        recording: rr.RecordingStream | None = None,
    ) -> np.ndarray:
        del video_source, timeline
        rr.log(str(video_log_path), rr.Image(np.zeros((24, 32, 3), dtype=np.uint8)), static=True, recording=recording)
        return np.arange(2, dtype=np.int64) * 100_000_000

    statuses: list[str] = list(
        run_video_pose_pipeline(
            SapiensVideoPoseConfig(
                video_path=video_path,
                rrd_path=tmp_path / "pose.rrd",
                tracking_backend="detr_per_frame",
                pose_backend="tensorrt",
                tensorrt_engine_path=tmp_path / "pose.trt",
                max_frames=2,
            ),
            frame_pose_fn=fake_frame_pose_fn,
            tensorrt_bbox_pose_fn=fake_bbox_pose_fn,
            detect_persons_fn=fake_detect_persons,
            log_video_fn=fake_log_video_fn,
            save_rrd=False,
        )
    )

    assert statuses[-1] == "Complete: 2 frame overlays streamed."
    assert len(detected_frames) == 2
    assert [bboxes.tolist() for bboxes in posed_bboxes] == [
        [[2.0, 3.0, 20.0, 22.0]],
        [[2.0, 3.0, 20.0, 22.0]],
    ]


def test_tensorrt_backend_uses_explicit_tensorrt_bbox_pose_fn(tmp_path: Path) -> None:
    image_rgb: UInt8[ndarray, "h w 3"] = np.zeros((8, 8, 3), dtype=np.uint8)
    bboxes: Float32[ndarray, "n 4"] = np.asarray([[0.0, 0.0, 7.0, 7.0]], dtype=np.float32)
    calls: list[str] = []

    def pytorch_bbox_pose_fn(*_args: object, **_kwargs: object) -> PoseEstimation:
        raise AssertionError("TensorRT routing should not call the PyTorch bbox pose override.")

    def tensorrt_bbox_pose_fn(
        image: UInt8[ndarray, "h w 3"],
        input_bboxes: Float32[ndarray, "n 4"],
        **_kwargs: object,
    ) -> PoseEstimation:
        assert image.shape == image_rgb.shape
        calls.append("tensorrt")
        keypoints: Float32[ndarray, "sapiens_k 2"] = np.zeros((308, 2), dtype=np.float32)
        scores: Float32[ndarray, "sapiens_k"] = np.ones((308,), dtype=np.float32)
        return PoseEstimation(bboxes=input_bboxes, keypoints=[keypoints], scores=[scores])

    effective_bbox_pose_fn = _make_effective_bbox_pose_fn(
        SapiensVideoPoseConfig(
            video_path=tmp_path / "input.mp4",
            rrd_path=tmp_path / "pose.rrd",
            pose_backend="tensorrt",
        ),
        pytorch_bbox_pose_fn,
        tensorrt_bbox_pose_fn=tensorrt_bbox_pose_fn,
    )

    result: PoseEstimation = effective_bbox_pose_fn(image_rgb, bboxes, model_size="0.4B", device="cpu")

    assert calls == ["tensorrt"]
    assert np.array_equal(result.bboxes, bboxes)


def test_default_tensorrt_engine_path_uses_stable_cache(monkeypatch: Any, tmp_path: Path) -> None:
    cache_root: Path = tmp_path / "cache"
    monkeypatch.delenv(DEFAULT_TENSORRT_ENGINE_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))

    engine_path: Path = Path(_default_tensorrt_engine_path())

    assert engine_path == cache_root / "sapiens2-pose" / "tensorrt" / "sapiens2_0_4b_pose_static_b1_bf16_current_static_graph.trt"

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    home_engine_path: Path = Path(_default_tensorrt_engine_path())

    assert home_engine_path == Path.home() / ".cache" / "sapiens2-pose" / "tensorrt" / "sapiens2_0_4b_pose_static_b1_bf16_current_static_graph.trt"


def test_default_tensorrt_engine_path_honors_explicit_env_var(monkeypatch: Any, tmp_path: Path) -> None:
    engine_path: Path = tmp_path / "custom.trt"
    monkeypatch.setenv(DEFAULT_TENSORRT_ENGINE_ENV_VAR, str(engine_path))

    assert _default_tensorrt_engine_path() == str(engine_path)


def test_format_video_inference_status_reports_backend_and_elapsed_time() -> None:
    status: str = _format_video_inference_status(
        status="Complete: 3 frame overlays streamed.",
        backend_label="TensorRT",
        elapsed_seconds=1.25,
    )

    assert status == "Complete: 3 frame overlays streamed. TensorRT inference took 1.25s for the whole video."


def _write_test_video(video_path: Path, *, num_frames: int) -> None:
    writer: cv2.VideoWriter = cv2.VideoWriter(
        str(video_path),
        int(cast(Any, cv2).VideoWriter_fourcc(*"mp4v")),
        10.0,
        (32, 24),
    )
    try:
        for frame_idx in range(num_frames):
            frame_bgr: UInt8[ndarray, "h w 3"] = np.full((24, 32, 3), 32 * frame_idx, dtype=np.uint8)
            writer.write(frame_bgr)
    finally:
        writer.release()
