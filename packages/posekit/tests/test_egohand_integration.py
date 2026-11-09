"""Released-weight EgoHand checks; no synthetic weights or network downloads."""

import gc
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("warp", reason="EgoHand needs the posekit-hands Pixi environment (posekit-egohand feature)")
pytest.importorskip("soma", reason="EgoHand needs the SOMA model package in the posekit-hands Pixi environment")

from jaxtyping import Float64
from numpy import ndarray
from simplecv.rrd_query_utils import first_valid_value

from posekit.cameras import CalibratedCamera
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig
from posekit.models.egohand_fast import EgoHandFastConfig, EgoHandStack

pytestmark = pytest.mark.integration

CAMERA = "/world/rig_04/cam_00/pinhole"


def fast_config_for_device() -> EgoHandFastConfig:
    """Exercise the same fast executor with a precision supported by the test GPU."""
    if torch.cuda.get_device_capability()[0] < 10:
        return EgoHandStack.catalog(backbone_precision="bf16").pose
    return EgoHandStack.fast().pose


@pytest.fixture(autouse=True)
def release_cuda_test_models() -> Iterator[None]:
    """Release captured-graph cycles before the next released-model integration test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def camera_from_window(table: pa.Table) -> CalibratedCamera:
    """Read the catalog calibration; matrix components are column-major."""
    intrinsics: Float64[ndarray, "3 3"] = (
        np.asarray(first_valid_value(table[f"{CAMERA}:Pinhole:image_from_camera"]), dtype=np.float64).reshape(3, 3).T
    )
    distortion: Float64[ndarray, "14"] = np.asarray(
        first_valid_value(table[f"{CAMERA}:simplecv.components.DistortionCoefficients"]), dtype=np.float64
    ).reshape(14)
    resolution = first_valid_value(table[f"{CAMERA}:Pinhole:resolution"])
    return CalibratedCamera(intrinsics, distortion, int(resolution[0]), int(resolution[1]))


def test_tensorrt_detector_preserves_batched_anatomical_detections() -> None:
    """Changing the detector runtime preserves boxes, count and anatomical side."""
    from PIL import Image

    from posekit.runtimes import OnnxBackendConfig, TensorRtBackendConfig

    weights = Path("/tmp/egohand/weights/detector/hand_detector.onnx")
    image = Path("/tmp/egohand/frame-4.144.png")
    for asset in (weights, image):
        if not asset.is_file():
            pytest.skip(f"missing EgoHand detector fixture: {asset}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    frames = torch.from_numpy(np.asarray(Image.open(image).convert("RGB")).copy()).cuda().unsqueeze(0).repeat(3, 1, 1, 1)
    reference = EgoHandDetectorConfig(weights, backend=OnnxBackendConfig(max_batch_size=3)).setup()(frames)
    result = EgoHandDetectorConfig(
        weights, backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=4, cache_dir=Path("/tmp/egohand/engines"))
    ).setup()(frames)
    assert result.sides == reference.sides
    assert torch.equal(result.boxes.frame_indices, reference.boxes.frame_indices)
    low = torch.maximum(result.boxes.xyxy[:, :2], reference.boxes.xyxy[:, :2])
    high = torch.minimum(result.boxes.xyxy[:, 2:], reference.boxes.xyxy[:, 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    areas = (result.boxes.xyxy[:, 2:] - result.boxes.xyxy[:, :2]).prod(-1)
    reference_areas = (reference.boxes.xyxy[:, 2:] - reference.boxes.xyxy[:, :2]).prod(-1)
    assert torch.median(intersection / (areas + reference_areas - intersection)) >= 0.99


def test_released_models_return_aligned_hands() -> None:
    weights = Path("/tmp/egohand/weights")
    required = [weights / "egohand/egohand_s060000.safetensors", weights / "detector/hand_detector.onnx", weights / "soma/SOMAHand.npz"]
    required.extend(
        weights / name
        for name in (
            "soma/SOMA_neutral.npz",
            "soma/SOMA_procedural_transforms.json",
            "soma/SOMA_template_rig.usda",
            "soma/MHR/mhr_model_lod1.pt",
            "soma/MHR/base_body_lod1.obj",
            "soma/MHR/SOMA_wrap_lod1.obj",
        )
    )
    for asset in required:
        if not asset.is_file():
            pytest.skip(f"missing EgoHand asset: {asset}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    image = Path("/tmp/egohand/frame-4.144.png")
    if not image.is_file():
        pytest.skip(f"missing one-segment RGB fixture: {image}")
    from PIL import Image

    rgb = np.asarray(Image.open(image).convert("RGB"))
    frames = torch.from_numpy(rgb.copy()).unsqueeze(0).cuda()
    window = Path("/tmp/egohand/window.parquet")
    if not window.is_file():
        pytest.skip(f"missing one-segment calibration fixture: {window}")
    camera = camera_from_window(pq.read_table(window))
    detections = EgoHandDetectorConfig(required[1]).setup()(frames)
    assert detections.sides == ("left", "right")
    assert float(detections.boxes.xyxy[0, 0]) < float(detections.boxes.xyxy[1, 0])
    model = EgoHandConfig(required[0], weights / "soma").setup()
    result = model(frames, detections, camera)
    assert len(result.sides) > 0
    assert result.keypoints.xyz_camera is not None
    assert torch.isfinite(result.keypoints.xyz_camera).all()
    assert result.keypoints.xy.shape == (len(result.sides), 25, 2)
    assert len(result.sides) == len(result.keypoints.frame_indices)
    assert result.scale_parameters is not None
    assert result.scale_parameters.shape == (len(result.sides), 26)
    assert result.rotations.shape == (len(result.sides), 25, 3, 3)
    for xyz, xy in zip(result.keypoints.xyz_camera.cpu().numpy(), result.keypoints.xy.cpu().numpy(), strict=True):
        expected, _ = cv2.projectPoints(xyz.astype(np.float64), np.zeros(3), np.zeros(3), camera.intrinsics, camera.distortion)
        np.testing.assert_allclose(xy, expected[:, 0], atol=1e-3)


def test_accelerated_predictor_retains_mesh_and_joint_parity() -> None:
    """TensorRT, GPU crops and batched SOMA stay within the geometry guard."""

    from PIL import Image

    from posekit.runtimes import TensorRtBackendConfig

    root = Path("/tmp/egohand")
    required = [
        root / name
        for name in (
            "weights/egohand/egohand_s060000.safetensors",
            "weights/soma/SOMAHand.npz",
            "weights/soma/MHR/mhr_model_lod1.pt",
            "weights/detector/hand_detector.onnx",
            "dinov3-bf16.onnx",
            "window.parquet",
            "frame-4.144.png",
        )
    ]
    other_assets = [
        root / "weights/soma" / name
        for name in (
            "SOMA_neutral.npz",
            "SOMA_procedural_transforms.json",
            "SOMA_template_rig.usda",
            "MHR/base_body_lod1.obj",
            "MHR/SOMA_wrap_lod1.obj",
        )
    ]
    for asset in required + other_assets:
        if not asset.is_file():
            pytest.skip(f"missing accelerated EgoHand asset: {asset}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    frames = torch.from_numpy(np.asarray(Image.open(required[-1]).convert("RGB")).copy()).cuda().unsqueeze(0).repeat(2, 1, 1, 1)
    camera = camera_from_window(pq.read_table(required[-2]))
    boxes = EgoHandDetectorConfig(required[3]).setup()(frames)
    eager = EgoHandConfig(required[0], root / "weights/soma").setup()
    expected = eager(frames, boxes, camera)
    assert expected.keypoints.xyz_camera is not None
    reference = expected.keypoints.xyz_camera.cpu()
    pixels = expected.keypoints.xy.cpu()
    del eager
    gc.collect()
    torch.cuda.empty_cache()
    accelerated = EgoHandConfig(
        required[0],
        root / "weights/soma",
        batch_size=32,
        gpu_crops=True,
        backbone_onnx=required[4],
        backbone_backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=32, cache_dir=root / "engines"),
    ).setup()
    result = accelerated(frames, boxes, camera)
    assert result.keypoints.xyz_camera is not None
    assert result.sides == expected.sides
    assert result.scale_parameters is not None
    assert result.vertices is not None and expected.vertices is not None
    assert all(mesh.shape == original.shape and mesh.numel() > 0 for mesh, original in zip(result.vertices, expected.vertices, strict=True))
    errors = torch.linalg.vector_norm(result.keypoints.xyz_camera.cpu() - reference, dim=-1) * 1000.0
    assert errors.median() <= 1.0
    assert torch.quantile(errors, 0.99) <= 5.0
    assert torch.linalg.vector_norm(result.keypoints.xy.cpu() - pixels, dim=-1).median() <= 0.5


def test_tensorrt_backbone_setup_never_builds_the_eager_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a cached backbone engine, set-up neither builds nor loads the 3 GB PyTorch DINOv3."""
    from dinov3.models.vision_transformer import DinoVisionTransformer

    from posekit.runtimes import TensorRtBackendConfig

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("built the eager DINOv3 backbone")

    monkeypatch.setattr(DinoVisionTransformer, "__init__", refuse)

    root = Path("/tmp/egohand")
    weights, soma, onnx = root / "weights/egohand/egohand_s060000.safetensors", root / "weights/soma", root / "dinov3-bf16.onnx"
    for asset in (weights, soma / "SOMAHand.npz", onnx):
        if not asset.exists():
            pytest.skip(f"missing accelerated EgoHand asset: {asset}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    model = EgoHandConfig(
        weights, soma, backbone_onnx=onnx, backbone_backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=32, cache_dir=root / "engines")
    ).setup()
    names = [name for name, _ in model.predictor.model.named_parameters()]
    assert names, "the hand head is still loaded"
    assert not [name for name in names if name.startswith("backbone.")]


@pytest.mark.parametrize("fp16_trunk", [False, True])
def test_gpu_detector_preprocessing_preserves_boxes_and_sides(fp16_trunk: bool) -> None:
    """GPU input preparation keeps the detector's anatomical decisions and boxes."""
    from dataclasses import replace

    from torchcodec.decoders import VideoDecoder

    from posekit.apis.video_hands import EXAMPLES
    from posekit.models.egohand_fast import EgoHandStack

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    video: Path = EXAMPLES / "selfcap_oak_left.mp4"
    if not video.is_file():
        pytest.skip(f"missing detector parity fixture: {video}")
    config = EgoHandStack.selfcap().detector
    decoder = VideoDecoder(video, dimension_order="NHWC", device="cuda")
    frames = decoder.get_frames_at([0, 20, 40, 60, 80]).data.contiguous()
    detector = config.setup()
    expected = detector(frames)
    detector = replace(config, gpu_preprocessing=True, fp16_trunk=fp16_trunk).setup()
    result = detector(frames)
    assert result.sides == expected.sides
    assert torch.equal(result.boxes.frame_indices, expected.boxes.frame_indices)
    low = torch.maximum(result.boxes.xyxy[:, :2], expected.boxes.xyxy[:, :2])
    high = torch.minimum(result.boxes.xyxy[:, 2:], expected.boxes.xyxy[:, 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    area = (result.boxes.xyxy[:, 2:] - result.boxes.xyxy[:, :2]).prod(-1)
    reference_area = (expected.boxes.xyxy[:, 2:] - expected.boxes.xyxy[:, :2]).prod(-1)
    assert torch.all(intersection / (area + reference_area - intersection) >= 0.85)


def test_cuda_execution_retains_joints_meshes_and_soma_parameters() -> None:
    """Captured CUDA crops and execution preserve the reference reconstruction with the same bf16 networks."""
    from torchcodec.decoders import VideoDecoder

    from posekit.apis.video_hands import EXAMPLES
    from posekit.cameras import load_calibrated_camera
    from posekit.models.egohand_cuda_crops import CudaHandCropper
    from posekit.models.egohand_fast import EgoHandStack
    from posekit.models.egohand_soma import prepare_cuda_soma
    from posekit.models.egohand_tensor import EgoHandTensorExecutor

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    video: Path = EXAMPLES / "selfcap_oak_left.mp4"
    if not video.is_file():
        pytest.skip(f"missing EgoHand parity fixture: {video}")
    stack = EgoHandStack.selfcap()
    detector_config, config = stack.detector, stack.pose
    camera = load_calibrated_camera(EXAMPLES / "selfcap_oak_left.rrd")
    decoder = VideoDecoder(video, dimension_order="NHWC", device="cuda")
    frames = decoder.get_frames_at([0, 30, 80]).data.contiguous()
    detector = detector_config.setup()
    reference = config.setup()
    # A second copy: capture preparation changes its SOMA model in place.
    model = config.setup().predictor.model
    executor = EgoHandTensorExecutor(model, config.batch_size)
    prepare_cuda_soma(model.soma_hand)
    cropper = CudaHandCropper(camera, frames.device)
    with torch.inference_mode():
        for frame in frames:
            batch = frame.unsqueeze(0)
            detections = detector(batch)
            expected = reference(batch, detections, camera)
            result = executor(cropper.prepare(batch, detections, config.padding), detections.sides, cropper.project)
            assert expected.sides == detections.sides
            assert expected.vertices is not None and expected.keypoints.xyz_camera is not None and expected.scale_parameters is not None
            torch.testing.assert_close(result.joints, expected.keypoints.xyz_camera, atol=0.001, rtol=0.001)
            torch.testing.assert_close(result.vertices, expected.vertices, atol=0.001, rtol=0.001)
            torch.testing.assert_close(result.pixels, expected.keypoints.xy, atol=0.5, rtol=0.001)
            torch.testing.assert_close(result.rotations, expected.rotations, atol=0.001, rtol=0.001)
            torch.testing.assert_close(result.scale_parameters, expected.scale_parameters, atol=0.001, rtol=0.001)


def test_cuda_soma_preserves_changing_shapes_and_full_meshes() -> None:
    """Captured skinning must use each frame's identity and scales on both sides."""
    from posekit.models.egohand_fast import EgoHandStack
    from posekit.models.egohand_soma import prepare_cuda_soma

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    config = EgoHandStack.selfcap().pose
    model = config.setup().predictor.model.soma_hand
    generator = torch.Generator(device="cuda").manual_seed(72)
    poses = torch.randn((2, 25, 3), device="cuda", generator=generator) * 0.2
    identity = torch.randn((2, 5), device="cuda", generator=generator) * 0.7
    scales = torch.randn((2, 26), device="cuda", generator=generator) * 0.05
    size = torch.tensor([0.9, 1.1], device="cuda")
    with torch.inference_mode():
        expected = {
            side: model(poses, identity, side, global_scale=size, scale_params=scales)
            for side in ("left", "right")
        }
        prepare_cuda_soma(model)
        for side in ("left", "right"):
            result = model(poses, identity, side, global_scale=size, scale_params=scales)
            for field in ("joints", "vertices"):
                assert result[field].shape == expected[side][field].shape
                torch.testing.assert_close(result[field], expected[side][field], atol=0.001, rtol=0.001)


def test_exportable_hand_head_preserves_virtual_camera_conditioning() -> None:
    """The exported head includes the same camera rays and both learned heads."""
    from egohand.utils.ray_pe import compute_crop_ray_angles, fourier_encode_angles

    from posekit.models.egohand_fast import EgoHandStack
    from posekit.models.egohand_head import EgoHandHead

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    config = EgoHandStack.selfcap().pose
    model = config.setup().predictor.model
    features = torch.randn((2, 1280, 16, 16), device="cuda")
    focal = torch.tensor([180.0, 510.0], device="cuda")
    intrinsics = torch.eye(3, device="cuda").expand(2, 3, 3).clone()
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = focal
    intrinsics[:, :2, 2] = 127.5
    boxes = torch.tensor([[127.5, 127.5, 256.0]], device="cuda").expand(2, -1)
    cliff = torch.stack((torch.zeros_like(focal), torch.zeros_like(focal), 256.0 / focal), dim=-1)
    rays = fourier_encode_angles(compute_crop_ray_angles(boxes, intrinsics, grid_hw=(16, 16), crop_hw=(256, 256)))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        params, camera, _ = model.hand_head(features, cliff_cond=cliff, ray_feats=rays)
        direct = model.kp2d_head(features)
        result = EgoHandHead(model)(features, focal)
    expected = (torch.cat((params["wrist_orient"], params["hand_pose"]), dim=1), params["identity_coeffs"], params["scale_params"], camera, direct["mu"], direct["presence"])
    for actual, reference in zip(result, expected, strict=True):
        torch.testing.assert_close(actual, reference.float(), atol=0.005, rtol=0.005)


def test_fast_role_chunks_large_frame_batches_without_losing_hand_alignment() -> None:
    """Offline callers can pass more hands than one TensorRT execution profile holds."""
    from torchcodec.decoders import VideoDecoder

    from posekit.apis.video_hands import EXAMPLES
    from posekit.cameras import load_calibrated_camera
    from posekit.models.egohand_fast import EgoHandStack
    from posekit.predictions import BoxDetections, HandDetections

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    video: Path = EXAMPLES / "selfcap_oak_left.mp4"
    if not video.is_file():
        pytest.skip(f"missing EgoHand parity fixture: {video}")
    detector_config = EgoHandStack.fast().detector
    pose_config: EgoHandFastConfig = fast_config_for_device()
    frames = VideoDecoder(video, device="cuda", dimension_order="NHWC")[0:1]
    camera = load_calibrated_camera(EXAMPLES / "selfcap_oak_left.rrd")
    detector, pose = detector_config.setup(), pose_config.setup()
    with torch.inference_mode():
        detections = detector(frames)
        expected = pose(frames, detections, camera)
        count: int = detections.num_detections
        assert count == 2
        large = HandDetections(BoxDetections(detections.boxes.xyxy.repeat(33, 1), detections.boxes.scores.repeat(33),
                                            torch.arange(33, device="cuda").repeat_interleave(count)), detections.sides * 33)
        result = pose(frames.repeat(33, 1, 1, 1), large, camera)
    assert result.sides == large.sides
    assert torch.equal(result.keypoints.frame_indices, large.boxes.frame_indices)
    assert result.vertices is not None and expected.vertices is not None
    assert result.keypoints.xyz_camera is not None and expected.keypoints.xyz_camera is not None
    torch.testing.assert_close(result.keypoints.xyz_camera, expected.keypoints.xyz_camera.repeat(33, 1, 1), atol=0.005, rtol=0.005)
    torch.testing.assert_close(result.vertices, expected.vertices.repeat(33, 1, 1), atol=0.005, rtol=0.005)


def test_fast_role_graphs_depend_on_hand_counts_not_side_order() -> None:
    """Reordered hands of one batch size replay the same graphs and return the same hands."""
    from torchcodec.decoders import VideoDecoder

    from posekit.apis.video_hands import EXAMPLES
    from posekit.cameras import load_calibrated_camera
    from posekit.models.egohand_fast import EgoHandStack
    from posekit.predictions import BoxDetections, HandDetections

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    video: Path = EXAMPLES / "selfcap_oak_left.mp4"
    if not video.is_file():
        pytest.skip(f"missing EgoHand parity fixture: {video}")
    detector_config = EgoHandStack.fast().detector
    pose_config: EgoHandFastConfig = fast_config_for_device()
    frames = VideoDecoder(video, device="cuda", dimension_order="NHWC")[0:1].repeat(2, 1, 1, 1)
    camera = load_calibrated_camera(EXAMPLES / "selfcap_oak_left.rrd")
    detector, pose = detector_config.setup(), pose_config.setup()
    with torch.inference_mode():
        detections = detector(frames[:1])
        assert detections.sides == ("left", "right")
        boxes = detections.boxes.xyxy.repeat(2, 1)
        frame_indices = torch.tensor([0, 0, 1, 1], device="cuda")
        results = {}
        graph_ids: set[int] | None = None
        for order in ((0, 1, 2, 3), (1, 0, 3, 2), (0, 2, 1, 3), (3, 1, 2, 0)):
            rows = torch.tensor(order, device="cuda")
            hands = HandDetections(
                BoxDetections(boxes[rows], torch.ones(4, device="cuda"), frame_indices[rows]), tuple(("left", "right")[index % 2] for index in order)
            )
            result = pose(frames, hands, camera)
            current_graph_ids: set[int] = {id(graph) for graph in pose.executor.graphs.values()}
            if graph_ids is None:
                graph_ids = current_graph_ids
            else:
                assert current_graph_ids == graph_ids
            assert result.sides == hands.sides
            assert result.keypoints.xyz_camera is not None and result.vertices is not None
            results[order] = (result.keypoints.xyz_camera[torch.argsort(rows)], result.vertices[torch.argsort(rows)])
    for xyz, vertices in results.values():
        torch.testing.assert_close(xyz, results[(0, 1, 2, 3)][0], atol=0.001, rtol=0.0)
        torch.testing.assert_close(vertices, results[(0, 1, 2, 3)][1], atol=0.001, rtol=0.0)
    # A two-hand profile needs separate all-left and all-right chunks.
    assert len(pose.executor.graphs) == (2 if pose_config.batch_size < 4 else 1)
    assert [len(cropper.graphs) for cropper in pose.croppers.values()] == [1]


def test_stage_times_are_recorded_only_on_request() -> None:
    """Without profile_stages neither predictor records stage times; with it, both report the benchmark's stages."""
    from torchcodec.decoders import VideoDecoder

    from posekit.apis.video_hands import EXAMPLES
    from posekit.cameras import load_calibrated_camera
    from posekit.models.egohand_fast import EgoHandStack

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    video: Path = EXAMPLES / "selfcap_oak_left.mp4"
    if not video.is_file():
        pytest.skip(f"missing EgoHand parity fixture: {video}")
    frames = VideoDecoder(video, device="cuda", dimension_order="NHWC")[0:1]
    camera = load_calibrated_camera(EXAMPLES / "selfcap_oak_left.rrd")
    stack = EgoHandStack.selfcap()
    detector_config, reference_config = stack.detector, stack.pose
    detections = detector_config.setup()(frames)
    assert reference_config.backbone_backend is not None
    reference_config = replace(reference_config, backbone_backend=replace(reference_config.backbone_backend, max_batch_size=2, opt_batch_size=2))
    for config, stages in (
        (reference_config, {"preprocessing", "batch_assembly", "model_forward", "post_forward", "backbone", "head", "soma"}),
        (fast_config_for_device(), {"preprocessing", "backbone", "head", "soma"}),
    ):
        pose = config.setup()
        with torch.inference_mode():
            pose(frames, detections, camera)
            assert pose.stage_ms == {}
            pose.profile_stages = True
            pose(frames, detections, camera)
        assert set(pose.stage_ms) == stages
        assert all(len(values) == 1 and values[0] > 0.0 for values in pose.stage_ms.values())
        del pose
        gc.collect()
        torch.cuda.empty_cache()
