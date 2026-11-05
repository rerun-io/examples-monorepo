"""The fast body role keeps complete outputs and reuses captures as input values change."""

from dataclasses import fields
from pathlib import Path

import pytest
import torch

pytest.importorskip("yacs", reason="SAM 3D Body requires the posekit-body environment")

from huggingface_hub import try_to_load_from_cache
from torchvision.io import ImageReadMode, decode_image

from posekit.models.sam3d_body import SAM3D_BODY_REVISION, Sam3dBodyConfig, Sam3dBodyMode
from posekit.predictions import BoxDetections


@pytest.mark.integration
@pytest.mark.parametrize("mode,gpu_crops,compile_backbone", [
    ("body", False, False), ("full", False, False),
    ("body", True, False), ("full", True, False), ("full", True, True),
])
def test_fast_body_reuses_graphs_and_preserves_outputs(mode: Sam3dBodyMode, gpu_crops: bool, compile_backbone: bool) -> None:
    """Changed pixels and boxes reuse the graph while matching the eager role."""
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig

    checkpoint = try_to_load_from_cache("facebook/sam-3d-body-dinov3", "model.ckpt", revision=SAM3D_BODY_REVISION)
    image: Path = Path("/mnt/nas/ssd-8tb/posekit/goldens/sam3d-body-v1/inputs/Planche.jpg")
    if not torch.cuda.is_available() or not isinstance(checkpoint, str) or not image.is_file():
        pytest.skip("requires CUDA, cached SAM 3D Body model.ckpt and the Planche.jpg golden input")
    reference = Sam3dBodyConfig(mode=mode, checkpoint_dir=Path(checkpoint).parent).setup()
    fast = Sam3dBodyFastConfig(
        mode=mode, checkpoint_dir=Path(checkpoint).parent, gpu_preprocessing=gpu_crops,
        backbone_onnx=None, decoder_onnx_dir=None, compile_torch_backbone=compile_backbone,
    ).setup()
    frames = decode_image(str(image), mode=ImageReadMode.RGB).permute(1, 2, 0).contiguous().cuda()[None]
    height, width = frames.shape[1:3]
    boxes = BoxDetections(
        torch.tensor([[0.0, 0.0, float(width), float(height)]], device="cuda"),
        torch.ones(1, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"),
    )
    with torch.inference_mode():
        mesh_tolerance: float = 0.005 if gpu_crops else 0.0005
        pixel_tolerance: float = 3.0 if gpu_crops else 0.5
        # The released model's first full call differs slightly from later calls.
        reference(frames, boxes)
        fast(frames, boxes)
        expected = reference(frames, boxes)
        actual = fast(frames, boxes)
        torch.testing.assert_close(actual.vertices, expected.vertices, atol=mesh_tolerance, rtol=0.0)
        # Tiny geometry changes can move the quantized OpenCV hand crop sampler;
        # this remains stricter than the separate corpus accuracy bands.
        torch.testing.assert_close(actual.keypoints.xy, expected.keypoints.xy, atol=pixel_tolerance, rtol=0.0)
        count: int = fast.graph_count
        assert count > 0
        changed = frames.flip(2)
        expected = reference(changed, boxes)
        actual = fast(changed, boxes)
        assert fast.graph_count == count
        torch.testing.assert_close(actual.vertices, expected.vertices, atol=mesh_tolerance, rtol=0.0)
        torch.testing.assert_close(actual.keypoints.xy, expected.keypoints.xy, atol=pixel_tolerance, rtol=0.0)
        assert actual.vertices.is_cuda and actual.translation.is_cuda


@pytest.mark.integration
def test_fast_body_preserves_per_frame_prompt_rules_in_mixed_batch() -> None:
    """Two people in one frame retain dummy prompts; a separate single person removes them."""
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig

    checkpoint = try_to_load_from_cache("facebook/sam-3d-body-dinov3", "model.ckpt", revision=SAM3D_BODY_REVISION)
    image = Path("/mnt/nas/ssd-8tb/posekit/goldens/sam3d-body-v1/inputs/Planche.jpg")
    if not torch.cuda.is_available() or not isinstance(checkpoint, str) or not image.is_file():
        pytest.skip("requires CUDA, cached SAM 3D Body model.ckpt and the Planche.jpg golden input")
    pixels = decode_image(str(image), mode=ImageReadMode.RGB).permute(1, 2, 0).contiguous().cuda()
    frames = torch.stack((pixels, pixels.flip(1)))
    height, width = pixels.shape[:2]
    boxes = BoxDetections(
        torch.tensor([[0.0, 0.0, float(width), float(height)]], device="cuda").repeat(3, 1),
        torch.ones(3, device="cuda"), torch.tensor([0, 0, 1], device="cuda"),
    )
    reference = Sam3dBodyConfig(checkpoint_dir=Path(checkpoint).parent).setup()
    fast = Sam3dBodyFastConfig(checkpoint_dir=Path(checkpoint).parent, gpu_preprocessing=True, backbone_onnx=None, decoder_onnx_dir=None).setup()
    with torch.inference_mode():
        reference(frames, boxes)
        fast(frames, boxes)
        expected = reference(frames, boxes)
        actual = fast(frames, boxes)
    torch.testing.assert_close(actual.keypoints.frame_indices, boxes.frame_indices)
    torch.testing.assert_close(actual.keypoints.xy, expected.keypoints.xy, atol=3.0, rtol=0.0)
    torch.testing.assert_close(actual.vertices, expected.vertices, atol=0.006, rtol=0.0)


@pytest.mark.integration
def test_fast_body_exposes_differentiable_mhr_for_multiview() -> None:
    """The public geometry interface retains gradients for multiview fitting."""
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig

    checkpoint = try_to_load_from_cache("facebook/sam-3d-body-dinov3", "model.ckpt", revision=SAM3D_BODY_REVISION)
    if not torch.cuda.is_available() or not isinstance(checkpoint, str):
        pytest.skip("requires CUDA and the cached SAM 3D Body model.ckpt")
    fast = Sam3dBodyFastConfig(checkpoint_dir=Path(checkpoint).parent, backbone_onnx=None, decoder_onnx_dir=None).setup()
    pose = torch.full((1, 133), 0.01, device="cuda", requires_grad=True)
    rotation = torch.full((1, 3), 0.01, device="cuda", requires_grad=True)
    result = fast.mhr_model.mhr_forward(
        torch.zeros(1, 3, device="cuda"), rotation, pose,
        torch.zeros(1, 108, device="cuda"), torch.zeros(1, 28, device="cuda"),
        torch.zeros(1, 45, device="cuda"), return_keypoints=True,
    )
    assert isinstance(result, tuple)
    assert result[0].shape == (1, 18439, 3)
    assert result[1].shape == (1, 308, 3)
    result[1][:, :70].square().mean().backward()
    for value in (pose, rotation):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().max() > 0


@pytest.mark.integration
def test_calibrated_body_reuses_graph_after_batch_and_camera_changes() -> None:
    """Returning to a previous batch shape keeps its outputs after another camera/batch ran."""
    import numpy as np

    from posekit.cameras import CalibratedCamera
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
    from posekit.models.sam3d_body_quantization import body_backbone_path, load_body_calibration

    config = Sam3dBodyFastConfig()
    image = Path("/mnt/nas/ssd-8tb/posekit/goldens/sam3d-body-v1/inputs/Planche.jpg")
    assert config.backbone_onnx is not None
    artifact = body_backbone_path(config.backbone_onnx, load_body_calibration(config.calibration), config.fp8_tail_blocks,
                                  plan=config.backbone_plan, source_version=config.backbone_source_onnx.name)
    if not torch.cuda.is_available() or not artifact.is_file() or not image.is_file():
        pytest.skip("requires CUDA, cached calibrated SAM body backbone and Planche.jpg")
    fast = config.setup()
    pixels = decode_image(str(image), mode=ImageReadMode.RGB).permute(1, 2, 0).contiguous().cuda()
    height, width = pixels.shape[:2]
    intrinsics = np.array([[900.0, 0.0, width / 2], [0.0, 900.0, height / 2], [0.0, 0.0, 1.0]], dtype=np.float64)
    camera = CalibratedCamera(intrinsics, np.zeros(0, dtype=np.float64), width, height, lens="PINHOLE")
    boxes = BoxDetections(
        torch.tensor([[0.0, 0.0, float(width), float(height)]], device="cuda"),
        torch.ones(1, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"),
        torch.ones((1, height, width), dtype=torch.bool, device="cuda"),
    )
    with torch.inference_mode():
        expected = fast(pixels[None], boxes, camera)
        retained = [getattr(expected, item.name) for item in fields(expected) if isinstance(getattr(expected, item.name), torch.Tensor)]
        retained += [getattr(expected.keypoints, item.name) for item in fields(expected.keypoints) if isinstance(getattr(expected.keypoints, item.name), torch.Tensor)]
        snapshots = [value.clone() for value in retained]
        first_count = fast.graph_count
        assert boxes.masks is not None
        paired = BoxDetections(boxes.xyxy.repeat(2, 1), boxes.scores.repeat(2), torch.arange(2, device="cuda"), boxes.masks.repeat(2, 1, 1))
        fast(torch.stack((pixels, pixels.flip(1))), paired, camera)
        assert fast.graph_count > first_count
        count = fast.graph_count
        shifted_intrinsics = intrinsics.copy()
        shifted_intrinsics[0, 2] += 12.0
        shifted_camera = CalibratedCamera(shifted_intrinsics, np.zeros(0, dtype=np.float64), width, height, lens="PINHOLE")
        shifted = fast(pixels[None], boxes, shifted_camera)
        assert fast.graph_count == count
        assert not torch.allclose(shifted.translation, expected.translation)
        actual = fast(pixels[None], boxes, camera)
        assert fast.graph_count == count
        torch.testing.assert_close(actual.vertices, expected.vertices, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(actual.keypoints.xy, expected.keypoints.xy, atol=1e-4, rtol=0.0)
        for retained_value, snapshot in zip(retained, snapshots, strict=True):
            torch.testing.assert_close(retained_value, snapshot, atol=0.0, rtol=0.0)


@pytest.mark.integration
def test_fast_profiling_is_scoped_and_reuses_its_capture() -> None:
    """Profiling produces every fast stage without leaving timing hooks on ordinary inference."""
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig

    checkpoint = try_to_load_from_cache("facebook/sam-3d-body-dinov3", "model.ckpt", revision=SAM3D_BODY_REVISION)
    image = Path("/mnt/nas/ssd-8tb/posekit/goldens/sam3d-body-v1/inputs/Planche.jpg")
    if not torch.cuda.is_available() or not isinstance(checkpoint, str) or not image.is_file():
        pytest.skip("requires CUDA, cached SAM body checkpoint and Planche.jpg")
    fast = Sam3dBodyFastConfig().setup()
    model = fast.estimator.model
    assert not model.backbone._forward_hooks
    assert "mhr_forward" not in vars(model.head_pose)
    pixels = decode_image(str(image), mode=ImageReadMode.RGB).permute(1, 2, 0).contiguous().cuda()
    height, width = pixels.shape[:2]
    boxes = BoxDetections(torch.tensor([[0.0, 0.0, float(width), float(height)]], device="cuda"),
                          torch.ones(1, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"))
    with torch.inference_mode():
        ordinary = fast(pixels[None], boxes)
        fast.profile_stages = True
        fast(pixels[None], boxes)
        count = fast.graph_count
        fast.stage_ms.clear()
        profiled = fast(pixels[None], boxes)
        assert fast.graph_count == count
        for stage in ("backbone_body", "backbone_hands", "decoder_body", "decoder_hands", "decoder_prompt", "mhr", "other", "total"):
            assert stage in fast.stage_ms and sum(fast.stage_ms[stage]) >= 0.0
        assert not model.backbone._forward_hooks
        assert "mhr_forward" not in vars(model.head_pose)
        fast.profile_stages = False
        fast.stage_ms.clear()
        fast(pixels[None], boxes)
        assert not fast.stage_ms
        torch.testing.assert_close(profiled.vertices, ordinary.vertices, atol=0.005, rtol=0.0)


@pytest.mark.integration
def test_reduced_hand_geometry_preserves_every_decoder_layer() -> None:
    """Removing unused mesh support preserves hand keypoints and every layer's tokens."""
    import numpy as np
    from torch import Tensor, nn

    from posekit.models.sam3d_body_capture import DecoderOutput
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
    from posekit.models.sam3d_body_head import CaptureMhrHead
    from posekit.models.sam3d_body_mhr import BodySkinning

    checkpoint = try_to_load_from_cache("facebook/sam-3d-body-dinov3", "model.ckpt", revision=SAM3D_BODY_REVISION)
    asset = try_to_load_from_cache("facebook/sam-3d-body-dinov3", "assets/mhr_model.pt", revision=SAM3D_BODY_REVISION)
    image = Path("/mnt/nas/ssd-8tb/posekit/goldens/sam3d-body-v1/inputs/Planche.jpg")
    if not torch.cuda.is_available() or not isinstance(checkpoint, str) or not isinstance(asset, str) or not image.is_file():
        pytest.skip("requires CUDA, cached SAM body checkpoint/MHR asset and Planche.jpg")
    fast = Sam3dBodyFastConfig(
        checkpoint_dir=Path(checkpoint).parent, backbone_onnx=None, decoder_onnx_dir=None, capture_pipeline=False,
    ).setup()
    # Observe each invocation directly; graph replay does not call Python hooks.
    fast.body_decoder.capture = False
    fast.hand_decoder.capture = False
    model = fast.estimator.model
    head = model.head_pose_hand
    assert isinstance(head, CaptureMhrHead)
    source = torch.jit.load(asset, map_location="cuda")
    points: list[Tensor] = []
    tokens: list[Tensor] = []

    def observe_points(module: nn.Module, args: tuple[Tensor | None, ...], output: dict[str, Tensor | np.ndarray | None]) -> None:
        value = output["pred_keypoints_3d"]
        assert isinstance(value, Tensor)
        points.append(value.clone())

    def observe_tokens(module: nn.Module, args: tuple[Tensor | None, ...], output: tuple[Tensor, Tensor]) -> None:
        tokens.append(output[0].clone())

    pixels = decode_image(str(image), mode=ImageReadMode.RGB).permute(1, 2, 0).contiguous().cuda()
    frames = torch.stack((pixels, pixels.flip(1)))
    height, width = pixels.shape[:2]
    boxes = BoxDetections(
        torch.tensor([[0.0, 0.0, float(width), float(height)]], device="cuda").repeat(2, 1),
        torch.ones(2, device="cuda"), torch.arange(2, device="cuda"),
    )
    # Freeze the actual hand decoder inputs once. Re-running the body and
    # backbone would add unrelated numerical variation to this parity check.
    decoder_inputs: list[tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]] = []
    original_forward = fast.hand_decoder.forward

    def remember_inputs(
        image_embeddings: Tensor, init_estimate: Tensor | None = None,
        keypoints: Tensor | None = None, prev_estimate: Tensor | None = None,
        condition_info: Tensor | None = None, batch: dict[str, Tensor] | None = None,
    ) -> DecoderOutput:
        assert init_estimate is None and prev_estimate is None
        assert keypoints is not None and condition_info is not None and batch is not None
        decoder_inputs.append((image_embeddings.clone(), keypoints.clone(), condition_info.clone(), {key: value.clone() for key, value in batch.items()}))
        return original_forward(image_embeddings, keypoints=keypoints, condition_info=condition_info, batch=batch)

    with torch.inference_mode():
        fast.hand_decoder.forward = remember_inputs
        fast(frames, boxes)
        fast.hand_decoder.forward = original_forward
        assert len(decoder_inputs) == 1
        features, prompts, condition, batch = decoder_inputs[0]
        model._initialize_batch(batch)
        model.body_batch_idx = []
        model.hand_batch_idx = list(range(len(features)))
        head.register_forward_hook(observe_points)
        for layer in model.decoder_hand.layers:
            layer.register_forward_hook(observe_tokens)
        full_vertices = torch.nonzero(head.keypoint_mapping[:70, :18439].ne(0).any(dim=0)).flatten()
        head.keypoint_mhr = BodySkinning(source, full_vertices)
        head.compact_mapping = torch.cat((head.keypoint_mapping[:70, full_vertices], head.keypoint_mapping[:70, 18439:]), 1)
        fast.hand_decoder.run(features, prompts, condition, batch, None)
        points.clear()
        tokens.clear()
        fast.hand_decoder.run(features, prompts, condition, batch, None)
        expected_points, expected_tokens = points.copy(), tokens.copy()
        points.clear()
        tokens.clear()
        hand_vertices = torch.nonzero(head.keypoint_mapping[21:42, :18439].ne(0).any(dim=0)).flatten()
        assert len(hand_vertices) < len(full_vertices)
        head.keypoint_mhr = BodySkinning(source, hand_vertices)
        head.compact_mapping = torch.cat((head.keypoint_mapping[:70, hand_vertices], head.keypoint_mapping[:70, 18439:]), 1)
        fast.hand_decoder.run(features, prompts, condition, batch, None)
    assert len(tokens) == len(expected_tokens) == len(model.decoder_hand.layers)
    assert len(points) == len(expected_points) >= len(tokens)
    for actual, expected in zip(points, expected_points, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for actual, expected in zip(tokens, expected_tokens, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-6)
