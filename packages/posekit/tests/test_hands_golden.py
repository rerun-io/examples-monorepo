"""Ported EgoHand and WiLoR against their pre-port outputs (main a6ff6f14c) on the same SelfCap OAK frames.

The references live on the NAS (README beside them); every array is compared with a tight band, so a
port that changes crops, numerics or row order fails here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("warp", reason="EgoHand needs the posekit-hands Pixi environment (posekit-egohand feature)")
pytest.importorskip("ultralytics", reason="WiLoR needs the posekit-hands Pixi environment (posekit-wilor feature)")

from jaxtyping import UInt8
from rerun.chunk import RrdReader
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from posekit.cameras import CalibratedCamera, load_calibrated_camera
from posekit.predictions import BoxDetections, HandDetections, HandPoses3d

pytestmark = pytest.mark.golden

GOLDENS: Path = Path("/mnt/nas/ssd-8tb/posekit/goldens/hands-v1")
EXAMPLES: Path = Path(__file__).resolve().parents[1] / "assets" / "examples"
SIDES: dict[str, int] = {"left": 0, "right": 1}


def read_golden(name: str, group: str) -> dict[str, np.ndarray]:
    """One ``/golden/<group>`` entity of a reference .rrd as named arrays."""
    path: Path = GOLDENS / name
    if not path.is_file():
        pytest.skip(f"missing pre-port golden: {path}")
    for chunk in RrdReader(path).stream():
        if chunk.entity_path == f"/golden/{group}":
            batch = chunk.to_record_batch()
            return {
                column: np.asarray(batch.column(column)[0].values).reshape(tuple(np.asarray(batch.column(f"{column}_shape")[0].values)))
                for column in batch.schema.names
                if not column.endswith("_shape") and not column.startswith("rerun.")
            }
    raise KeyError(group)


@pytest.fixture(scope="module")
def oak() -> tuple[UInt8[Tensor, "100 720 1280 3"], CalibratedCamera]:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    frames = VideoDecoder(EXAMPLES / "selfcap_oak_left.mp4", dimension_order="NHWC", device="cuda").get_frames_in_range(0, 100).data.contiguous()
    return frames, load_calibrated_camera(EXAMPLES / "selfcap_oak_left.json")


def assert_detections(actual: HandDetections, expected: dict[str, np.ndarray]) -> None:
    assert [SIDES[side] for side in actual.sides] == expected["sides"].tolist()
    np.testing.assert_array_equal(actual.boxes.frame_indices.cpu().numpy(), expected["frame_indices"])
    np.testing.assert_allclose(actual.boxes.xyxy_numpy(), expected["xyxy"], atol=1e-3, rtol=0)
    np.testing.assert_allclose(actual.boxes.scores.cpu().numpy(), expected["scores"], atol=1e-5, rtol=0)


def assert_hands(actual: HandPoses3d, expected: dict[str, np.ndarray], *, xyz_m: float, xy_px: float) -> None:
    assert actual.keypoints.xyz_camera is not None
    assert actual.vertices is not None
    assert [SIDES[side] for side in actual.sides] == expected["sides"].tolist()
    np.testing.assert_array_equal(actual.keypoints.frame_indices.cpu().numpy(), expected["frame_indices"])
    np.testing.assert_allclose(actual.keypoints.xyz_camera.cpu().numpy(), expected["xyz_camera"], atol=xyz_m, rtol=0)
    np.testing.assert_allclose(actual.keypoints.xy.cpu().numpy(), expected["xy"], atol=xy_px, rtol=0)
    np.testing.assert_allclose(actual.vertices.cpu().numpy(), expected["vertices"], atol=xyz_m, rtol=0)


def test_egohand_eager_matches_preport(oak) -> None:
    """ORT detector + eager predictor, the reference path."""
    from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig, EgoHandReconstruction

    frames, camera = oak
    with torch.inference_mode():
        detections: HandDetections = EgoHandDetectorConfig().setup()(frames)
        assert_detections(detections, read_golden("egohand_golden.rrd", "egohand_eager_detections"))
        result = EgoHandConfig().setup()(frames, detections, camera)
    assert isinstance(result, EgoHandReconstruction)
    expected = read_golden("egohand_golden.rrd", "egohand_eager_hands")
    assert_hands(result, expected, xyz_m=1e-5, xy_px=1e-3)
    np.testing.assert_allclose(result.rotations.cpu().numpy(), expected["rotations"], atol=1e-5, rtol=0)
    assert result.scale_parameters is not None
    np.testing.assert_allclose(result.scale_parameters.cpu().numpy(), expected["scale_parameters"], atol=1e-5, rtol=0)


def test_egohand_selfcap_tensorrt_matches_preport(oak) -> None:
    """SelfCap's configuration: TRT detector, GPU crops, TRT bf16 backbone, 8 frames per predictor call."""
    from posekit.models.egohand import selfcap_egohand_configs

    frames, camera = oak
    detector_config, predictor_config = selfcap_egohand_configs(None)
    with torch.inference_mode():
        detections: HandDetections = detector_config.setup()(frames)
        assert_detections(detections, read_golden("egohand_golden.rrd", "egohand_trt_detections"))
        predictor = predictor_config.setup()
        rows: dict[str, list[np.ndarray]] = {key: [] for key in ("frame_indices", "sides", "xyz_camera", "xy", "vertices")}
        for start in range(0, len(frames), 8):
            keep: list[int] = [i for i, frame in enumerate(detections.boxes.frame_indices.tolist()) if start <= frame < start + 8]
            chunk = HandDetections(
                BoxDetections(detections.boxes.xyxy[keep], detections.boxes.scores[keep], detections.boxes.frame_indices[keep] - start),
                tuple(detections.sides[i] for i in keep),
            )
            result = predictor(frames[start : start + 8], chunk, camera)
            assert result.keypoints.xyz_camera is not None and result.vertices is not None
            rows["frame_indices"].append(result.keypoints.frame_indices.cpu().numpy() + start)
            rows["sides"].append(np.asarray([SIDES[side] for side in result.sides]))
            rows["xyz_camera"].append(result.keypoints.xyz_camera.cpu().numpy())
            rows["xy"].append(result.keypoints.xy.cpu().numpy())
            rows["vertices"].append(result.vertices.cpu().numpy())
    expected = read_golden("egohand_golden.rrd", "egohand_trt_hands")
    for key, values in rows.items():
        np.testing.assert_allclose(np.concatenate(values), expected[key], atol=1e-5 if key != "xy" else 1e-3, rtol=0, err_msg=key)


def assert_band(actual: np.ndarray, expected: np.ndarray, *, median: float, p99: float, scale: float = 1.0) -> None:
    """Per-point Euclidean error (times ``scale``) within a median and a 99th-percentile band."""
    error: np.ndarray = np.linalg.norm(actual - expected, axis=-1) * scale
    assert float(np.median(error)) <= median, f"median {np.median(error):.4f} > {median}"
    assert float(np.quantile(error, 0.99)) <= p99, f"p99 {np.quantile(error, 0.99):.4f} > {p99}"


@pytest.mark.parametrize("backend", ["torch", "tensorrt"])
@pytest.mark.parametrize("clip", ["oak", "video"])
def test_wilor_matches_preport(oak, backend: str, clip: str) -> None:
    """YOLO hand detector + WiLoR on both of today's paths (torch reference, static TRT engines).

    WiLoR runs in fp16. In one process the port reproduces main's network bit for bit; across processes
    and the environment change (torch 2.12.0 -> 2.12.1, OpenCV 4.13 -> 5.0) fp16 kernel choice moves
    single vertices by up to ~3.5 mm, so 3D and 2D outputs are held to median/p99 bands.
    """
    from posekit.models.wilor import WilorHandDetectorConfig, WilorHandPoseConfig, WilorHands
    from posekit.runtimes import TensorRtBackendConfig, TorchBackendConfig

    frames = oak[0] if clip == "oak" else (
        VideoDecoder(EXAMPLES / "hands_video.mp4", dimension_order="NHWC", device="cuda").get_frames_in_range(0, 30).data.contiguous()
    )
    backend_config = TorchBackendConfig(autocast="fp16") if backend == "torch" else TensorRtBackendConfig()
    expected = read_golden("wilor_golden.rrd", f"wilor_{'torch' if backend == 'torch' else 'trt'}_{clip}")
    with torch.inference_mode():
        detections = WilorHandDetectorConfig(backend=backend_config).setup()(frames)
        result = WilorHandPoseConfig(backend=backend_config).setup()(frames, detections)
    assert isinstance(result, WilorHands)
    # NMS orders equal-score boxes arbitrarily (frame 56 of the OAK clip ties at 0.815), so compare rows in (frame, side) order.
    sides = np.asarray([SIDES[side] for side in result.sides])
    frames_of = result.keypoints.frame_indices.cpu().numpy()
    mine = np.lexsort((sides, frames_of))
    theirs = np.lexsort((expected["sides"], expected["frame_indices"]))
    expected = {key: value[theirs] for key, value in expected.items()}
    np.testing.assert_array_equal(sides[mine], expected["sides"])
    np.testing.assert_array_equal(frames_of[mine], expected["frame_indices"])
    np.testing.assert_allclose(detections.boxes.xyxy_numpy()[mine], expected["xyxy"], atol=0.1, rtol=0)
    assert result.vertices is not None and result.keypoints.xyz_camera is not None
    # Camera-space outputs are WiLoR's wrist-relative MANO plus its full-image translation.
    translation = result.cam_t_full.cpu().numpy()[mine][:, None]
    assert_band(result.vertices.cpu().numpy()[mine] - translation, expected["vertices"], median=0.25, p99=3.0, scale=1000.0)
    assert_band(result.keypoints.xyz_camera.cpu().numpy()[mine] - translation, expected["keypoints_3d"], median=0.25, p99=3.0, scale=1000.0)
    assert_band(result.keypoints.xy.cpu().numpy()[mine], expected["keypoints_2d"], median=0.4, p99=1.0)
    depth_error = np.abs(result.cam_t_full.cpu().numpy()[mine][:, 2] / expected["cam_t_full"][:, 2] - 1.0)
    assert float(np.median(depth_error)) <= 1e-3 and float(depth_error.max()) <= 2e-2
    np.testing.assert_allclose(result.betas.cpu().numpy()[mine], expected["betas"], atol=2e-2, rtol=0)
