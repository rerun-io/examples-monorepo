"""Ported SAM 3D Body role against its references on fixed inputs (boxes, masks, cameras, pixels; NAS, README there).

- ``sam3d_body_posekit_env.rrd``: upstream's estimator API (the vendored, otherwise unported model) in this environment.
  Every key must match within atol = max(3 x its recorded run-to-run deviation, floor); the model varies up to 0.7 mm
  on one input (three people, no masks) and below 1e-6 on the others. This catches any change the port makes.
- ``sam3d_body_golden.rrd``: the pre-port package in its own environment (packages/sam3d-body, ef5ce8bdb). The vendored
  code reproduces it there; here torch/cuDNN/OpenCV versions differ, so the role must stay inside a per-point
  median/p99 band of it, set to twice the drift measured between the two references.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("yacs", reason="SAM 3D Body needs the posekit-body Pixi environment (posekit-sam3d-body feature)")

import cv2
from jaxtyping import UInt8
from rerun.chunk import RrdReader
from torch import Tensor

from posekit.cameras import CalibratedCamera
from posekit.models.sam3d_body import Sam3dBodyConfig, Sam3dBodyMode, Sam3dBodyPredictor, Sam3dBodyReconstruction
from posekit.predictions import BoxDetections

pytestmark = pytest.mark.golden

GOLDENS: Path = Path("/mnt/nas/ssd-8tb/posekit/goldens/sam3d-body-v1")
EXAMPLES: Path = Path(__file__).resolve().parents[1] / "assets" / "examples"
NOISE_FACTOR: float = 3.0
"""Tolerance as a multiple of the largest deviation the unported model showed between its own runs."""
FLOOR_M: float = 1e-5
"""Smallest tolerance for metric and parameter keys."""
FLOOR_PX: float = 1e-3
"""Smallest tolerance for pixel keys."""
PIXEL_KEYS: frozenset[str] = frozenset({"pred_keypoints_2d", "bbox", "lhand_bbox", "rhand_bbox", "focal_length"})


PREPORT_BAND: dict[str, tuple[float, float, float, float]] = {
    "planche_body": (0.8, 2.7, 0.3, 0.8),
    "planche_full": (0.9, 3.3, 0.5, 1.1),
    "boxing_full_nocam": (1.9, 17.0, 0.6, 4.8),
    "video_full": (0.9, 35.0, 0.4, 47.0),
    "hocap_full": (1.2, 104.0, 0.2, 3.1),
}
"""Per case: vertex error median and p99 (mm), then 2D keypoint error median and p99 (px), versus the pre-port golden."""


def read_goldens(name: str = "sam3d_body_posekit_env.rrd") -> dict[str, dict[str, np.ndarray]]:
    """Every ``/golden/<group>`` entity of a reference .rrd as named arrays."""
    path: Path = GOLDENS / name
    if not path.is_file():
        pytest.skip(f"missing pre-port golden: {path}")
    groups: dict[str, dict[str, np.ndarray]] = {}
    for chunk in RrdReader(path).stream():
        if chunk.entity_path.startswith("/golden/"):
            batch = chunk.to_record_batch()
            groups[chunk.entity_path.removeprefix("/golden/")] = {
                column: np.asarray(batch.column(column)[0].values).reshape(tuple(np.asarray(batch.column(f"{column}_shape")[0].values)))
                for column in batch.schema.names
                if not column.endswith("_shape") and not column.startswith("rerun.")
            }
    return groups


@pytest.fixture(scope="module")
def goldens() -> dict[str, dict[str, np.ndarray]]:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    return read_goldens()


@pytest.fixture(scope="module")
def preport() -> dict[str, dict[str, np.ndarray]]:
    return read_goldens("sam3d_body_golden.rrd")


@pytest.fixture(scope="module")
def predictors(goldens) -> dict[Sam3dBodyMode, Sam3dBodyPredictor]:
    """Both modes, each warmed by one call: a model's first full-mode call differs from every later one by up to
    0.2 mm (later calls agree to 1e-7), and the references were recorded on warm models."""
    predictors: dict[Sam3dBodyMode, Sam3dBodyPredictor] = {"full": Sam3dBodyConfig(mode="full").setup(), "body": Sam3dBodyConfig(mode="body").setup()}
    inputs = goldens["boxing_inputs"]
    frames = read_image("Amir-Khan-Lamont-Peterson_2689582.jpg", inputs)
    with torch.inference_mode():
        for predictor in predictors.values():
            predictor(frames, detections(inputs, masks=False))
    return predictors


def sha256(rgb: np.ndarray) -> np.ndarray:
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).digest(), dtype=np.uint8)


def read_image(name: str, inputs: dict[str, np.ndarray]) -> UInt8[Tensor, "1 h w 3"]:
    path: Path = GOLDENS / "inputs" / name
    bgr: np.ndarray | None = cv2.imread(str(path))
    assert bgr is not None, path
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(sha256(rgb), inputs["rgb_sha256"], err_msg=f"{path} decodes to other pixels than at capture")
    return torch.from_numpy(rgb).cuda()[None]


def pinhole(intrinsics: np.ndarray, frames: Tensor) -> CalibratedCamera:
    return CalibratedCamera(intrinsics.astype(np.float64), np.zeros(0), int(frames.shape[2]), int(frames.shape[1]), lens="PINHOLE")


def detections(inputs: dict[str, np.ndarray], *, masks: bool, frame_indices: np.ndarray | None = None) -> BoxDetections:
    count: int = len(inputs["xyxy"])
    return BoxDetections(
        torch.from_numpy(inputs["xyxy"]).cuda(),
        torch.from_numpy(inputs["scores"]).cuda() if "scores" in inputs else torch.ones(count, device="cuda"),
        torch.from_numpy(np.zeros(count, np.int64) if frame_indices is None else frame_indices).cuda(),
        masks=torch.from_numpy(inputs["masks"] > 0).cuda() if masks else None,
    )


def assert_matches(actual: Sam3dBodyReconstruction, expected: dict[str, np.ndarray], noise: dict[str, np.ndarray]) -> None:
    """Every role output against the pre-port fields it is built from (camera-space ones add ``pred_cam_t``)."""

    def tolerance(*keys: str) -> float:
        floor: float = FLOOR_PX if keys[0] in PIXEL_KEYS else FLOOR_M
        return max(NOISE_FACTOR * sum(float(noise[key][0]) for key in keys), floor)

    translation: np.ndarray = expected["pred_cam_t"]
    pairs: list[tuple[str, Tensor | None, np.ndarray, tuple[str, ...]]] = [
        ("xyz_camera", actual.keypoints.xyz_camera, expected["pred_keypoints_3d"] + translation[:, None], ("pred_keypoints_3d", "pred_cam_t")),
        ("xy", actual.keypoints.xy, expected["pred_keypoints_2d"], ("pred_keypoints_2d",)),
        ("vertices", actual.vertices, expected["pred_vertices"] + translation[:, None], ("pred_vertices", "pred_cam_t")),
        ("joint_coordinates", actual.joint_coordinates, expected["pred_joint_coords"] + translation[:, None], ("pred_joint_coords", "pred_cam_t")),
        ("translation", actual.translation, translation, ("pred_cam_t",)),
        ("boxes", actual.boxes, expected["bbox"], ("bbox",)),
        ("focal_length", actual.focal_length, expected["focal_length"], ("focal_length",)),
        ("global_rotation", actual.global_rotation, expected["global_rot"], ("global_rot",)),
        ("body_pose", actual.body_pose, expected["body_pose_params"], ("body_pose_params",)),
        ("hand_pose", actual.hand_pose, expected["hand_pose_params"], ("hand_pose_params",)),
        ("scale", actual.scale, expected["scale_params"], ("scale_params",)),
        ("shape", actual.shape, expected["shape_params"], ("shape_params",)),
        ("expression", actual.expression, expected["expr_params"], ("expr_params",)),
        ("joint_rotations", actual.joint_rotations, expected["pred_global_rots"], ("pred_global_rots",)),
    ]
    if "lhand_bbox" in expected:
        pairs += [
            ("left_hand_boxes", actual.left_hand_boxes, expected["lhand_bbox"], ("lhand_bbox",)),
            ("right_hand_boxes", actual.right_hand_boxes, expected["rhand_bbox"], ("rhand_bbox",)),
        ]
    else:
        assert actual.left_hand_boxes is None and actual.right_hand_boxes is None
    np.testing.assert_array_equal(actual.keypoints.frame_indices.cpu().numpy(), expected["frame_indices"])
    for name, value, reference, keys in pairs:
        assert value is not None, name
        np.testing.assert_allclose(value.cpu().numpy(), reference, atol=tolerance(*keys), rtol=0, err_msg=name)


def assert_preport_band(actual: list[Sam3dBodyReconstruction], expected: dict[str, np.ndarray], case: str) -> None:
    """Per-point vertex and 2D keypoint errors against the pre-port package stay inside the case's median/p99 band."""
    vertex_median, vertex_p99, pixel_median, pixel_p99 = PREPORT_BAND[case]
    vertices: np.ndarray = np.concatenate([result.vertices.cpu().numpy() for result in actual])
    xy: np.ndarray = np.concatenate([result.keypoints.xy.cpu().numpy() for result in actual])
    vertices_mm: np.ndarray = np.linalg.norm(vertices - (expected["pred_vertices"] + expected["pred_cam_t"][:, None]), axis=-1) * 1000.0
    pixels: np.ndarray = np.linalg.norm(xy - expected["pred_keypoints_2d"], axis=-1)
    assert np.median(vertices_mm) <= vertex_median and np.percentile(vertices_mm, 99) <= vertex_p99, (np.median(vertices_mm), np.percentile(vertices_mm, 99))
    assert np.median(pixels) <= pixel_median and np.percentile(pixels, 99) <= pixel_p99, (np.median(pixels), np.percentile(pixels, 99))


@pytest.mark.parametrize("mode", ["body", "full"])
def test_image_with_masks_and_camera(goldens, preport, predictors, mode: Sam3dBodyMode) -> None:
    """Planche.jpg: SAM3 box + mask + score and MoGe-2 intrinsics, the CLI/Gradio path."""
    inputs = goldens["planche_inputs"]
    frames = read_image("Planche.jpg", inputs)
    with torch.inference_mode():
        result = predictors[mode](frames, detections(inputs, masks=True), pinhole(inputs["intrinsics"], frames))
    assert_matches(result, goldens[f"planche_{mode}"], goldens[f"planche_{mode}_noise"])
    assert_preport_band([result], preport[f"planche_{mode}"], f"planche_{mode}")


def test_three_people_without_masks_or_camera(goldens, preport, predictors) -> None:
    """Three boxes in one frame, no prompts, the default field of view."""
    inputs = goldens["boxing_inputs"]
    frames = read_image("Amir-Khan-Lamont-Peterson_2689582.jpg", inputs)
    with torch.inference_mode():
        result = predictors["full"](frames, detections(inputs, masks=False))
    assert_matches(result, goldens["boxing_full_nocam"], goldens["boxing_full_nocam_noise"])
    assert_preport_band([result], preport["boxing_full_nocam"], "boxing_full_nocam")


def test_video_clip(goldens, preport, predictors) -> None:
    """16 CUDA-decoded frames of the hands example video in one batch, one box per frame."""
    from torchcodec.decoders import VideoDecoder

    inputs = goldens["video_inputs"]
    frames = VideoDecoder(EXAMPLES / "hands_video.mp4", dimension_order="NHWC", device="cuda").get_frames_in_range(0, 16).data.contiguous()
    np.testing.assert_array_equal(np.stack([sha256(frame) for frame in frames.cpu().numpy()]), inputs["rgb_sha256"], err_msg="decoder output changed")
    with torch.inference_mode():
        result = predictors["full"](frames, detections(inputs, masks=False, frame_indices=inputs["frame_indices"]))
    assert_matches(result, goldens["video_full"], goldens["video_full_noise"])
    assert_preport_band([result], preport["video_full"], "video_full")


def test_multiview_hocap_sample(goldens, preport, predictors) -> None:
    """HO-Cap frame 0 seen by 8 calibrated cameras, one person with its SAM3 mask per view (the mv-body demo's call)."""
    inputs = goldens["hocap_inputs"]
    rows: list[Sam3dBodyReconstruction] = []
    with torch.inference_mode():
        for view in range(len(inputs["rgb"])):
            frames = torch.from_numpy(inputs["rgb"][view]).cuda()[None]
            view_inputs = {"xyxy": inputs["xyxy"][view : view + 1], "scores": inputs["scores"][view : view + 1], "masks": inputs["masks"][view : view + 1]}
            rows.append(predictors["full"](frames, detections(view_inputs, masks=True), pinhole(inputs["intrinsics"][view], frames)))
    expected = goldens["hocap_full"]
    noise = goldens["hocap_full_noise"]
    for view, result in enumerate(rows):
        assert_matches(result, {key: value[view : view + 1] if key != "frame_indices" else np.zeros(1, np.int64) for key, value in expected.items()}, noise)
    assert_preport_band(rows, preport["hocap_full"], "hocap_full")
