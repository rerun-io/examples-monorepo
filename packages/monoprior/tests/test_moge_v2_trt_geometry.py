"""Contract, parity, and convention tests for batched full MoGe v2 geometry."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import cv2
import numpy as np
import pytest
import torch
from jaxtyping import Bool, Float32, UInt8
from torch import Tensor

from monopriors.models.surface_normal.moge_v2_trt import Encoder

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_token_grid_hw_matches_pinned_aspect_buckets() -> None:
    """The 3,600-token budget produces the five fixed network grids."""
    from monopriors.models.monoprior.moge_v2_trt import ASPECT_BUCKETS, token_grid_hw

    expected_buckets: tuple[tuple[int, int], ...] = (
        (728, 966),
        (966, 728),
        (630, 1120),
        (1120, 630),
        (840, 840),
    )
    computed_buckets: tuple[tuple[int, int], ...] = (
        token_grid_hw(4.0 / 3.0, 3600),
        token_grid_hw(3.0 / 4.0, 3600),
        token_grid_hw(16.0 / 9.0, 3600),
        token_grid_hw(9.0 / 16.0, 3600),
        token_grid_hw(1.0, 3600),
    )

    assert computed_buckets == expected_buckets
    assert expected_buckets == ASPECT_BUCKETS


@pytest.mark.parametrize(
    ("input_hw", "expected_bucket"),
    (
        ((756, 1008), (728, 966)),
        ((1080, 1920), (630, 1120)),
        ((1920, 1080), (1120, 630)),
        ((800, 800), (840, 840)),
    ),
)
def test_select_aspect_bucket_uses_nearest_log_aspect(
    input_hw: tuple[int, int],
    expected_bucket: tuple[int, int],
) -> None:
    """Bucket selection snaps common input shapes to their nearest aspect."""
    from monopriors.models.monoprior.moge_v2_trt import ASPECT_BUCKETS, select_aspect_bucket

    selected_bucket: tuple[int, int] = select_aspect_bucket(input_hw, ASPECT_BUCKETS)

    assert selected_bucket == expected_bucket


def test_trt_predictor_loads_prebuilt_engine_without_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A supplied engine bypasses builds and forwards the CUDA-graph option."""
    import monopriors.models.monoprior.moge_v2_trt as moge_v2_trt

    class FakeRuntime:
        def __init__(self, engine_path: Path, *, use_cuda_graph: bool = False) -> None:
            self.engine_path: Path = engine_path
            self.use_cuda_graph: bool = use_cuda_graph

        def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
            return inputs

    def fail_export(*args: object, **kwargs: object) -> Path:
        raise AssertionError("prebuilt engine path must bypass ONNX export")

    engine_path: Path = tmp_path / "moge-geometry.engine"
    engine_path.write_bytes(b"engine")
    monkeypatch.setattr(moge_v2_trt, "export_moge_v2_geometry_onnx", fail_export)
    monkeypatch.setitem(sys.modules, "trtkit.tensorrt_runtime", SimpleNamespace(TensorRtRuntime=FakeRuntime))

    image_hw: tuple[int, int] = (728, 966)
    predictor: moge_v2_trt.MoGeV2TrtGeometryPredictor = moge_v2_trt.MoGeV2TrtGeometryPredictor(
        image_hw=image_hw,
        engine_path=engine_path,
        use_cuda_graph=True,
    )
    runtime: FakeRuntime = cast(FakeRuntime, predictor.runtimes[image_hw])

    assert runtime.engine_path == engine_path
    assert runtime.use_cuda_graph


def test_trt_predictor_requires_static_size_with_prebuilt_engine(tmp_path: Path) -> None:
    """A prebuilt engine cannot be mapped to a bucket without its static size."""
    from monopriors.models.monoprior.moge_v2_trt import MoGeV2TrtGeometryPredictor

    engine_path: Path = tmp_path / "moge-geometry.engine"
    engine_path.write_bytes(b"engine")

    with pytest.raises(ValueError, match="image_hw"):
        MoGeV2TrtGeometryPredictor(engine_path=engine_path)


def _synthetic_rgb_batch(batch_size: int, image_hw: tuple[int, int]) -> UInt8[Tensor, "b h w 3"]:
    """Create deterministic RGB gradients directly on CUDA.

    Args:
        batch_size: Number of frames to create.
        image_hw: Frame height and width.

    Returns:
        uint8 CUDA RGB frames shaped ``b h w 3``.
    """
    height: int = image_hw[0]
    width: int = image_hw[1]
    x_w: Float32[Tensor, "w"] = torch.linspace(0, 255, width, device="cuda", dtype=torch.float32)
    y_h: Float32[Tensor, "h"] = torch.linspace(0, 255, height, device="cuda", dtype=torch.float32)
    red_hw: Float32[Tensor, "h w"] = x_w[None, :].expand(height, width)
    green_hw: Float32[Tensor, "h w"] = y_h[:, None].expand(height, width)
    blue_hw: Float32[Tensor, "h w"] = (red_hw + green_hw) / 2.0
    rgb_hw3: UInt8[Tensor, "h w 3"] = torch.stack((red_hw, green_hw, blue_hw), dim=-1).to(torch.uint8)
    return rgb_hw3[None].expand(batch_size, -1, -1, -1).contiguous()


@requires_cuda
def test_torch_predictor_returns_batched_metric_geometry() -> None:
    """The torch twin returns owning CUDA tensors in the geometry contract."""
    from monopriors.models.monoprior.moge_v2_trt import MoGeV2GeometryOutput, MoGeV2TorchGeometryPredictor

    image_hw: tuple[int, int] = (56, 70)
    rgb_bhw3: UInt8[Tensor, "2 56 70 3"] = _synthetic_rgb_batch(2, image_hw)
    predictor: MoGeV2TorchGeometryPredictor = MoGeV2TorchGeometryPredictor(
        encoder="vits",
        image_hw=image_hw,
        resolution_level=0,
    )

    prediction: MoGeV2GeometryOutput = predictor(rgb_bhw3)

    assert prediction.depth_bhw.shape == (2, 56, 70)
    assert prediction.points_bhw3.shape == (2, 56, 70, 3)
    assert prediction.normals_bhw3.shape == (2, 56, 70, 3)
    assert prediction.mask_bhw.shape == (2, 56, 70)
    assert prediction.intrinsics_b33.shape == (2, 3, 3)
    assert prediction.metric_scale_b.shape == (2,)
    assert prediction.shift_b.shape == (2,)
    for output_tensor in prediction:
        assert output_tensor.dtype == torch.float32
        assert output_tensor.is_cuda
    assert torch.isfinite(prediction.depth_bhw).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction.normals_bhw3, dim=-1),
        torch.ones((2, 56, 70), device="cuda"),
        atol=2e-3,
        rtol=0.0,
    )
    assert prediction.mask_bhw.min().item() >= 0.0
    assert prediction.mask_bhw.max().item() <= 1.0
    valid_bhw: Bool[Tensor, "2 56 70"] = prediction.mask_bhw > 0.5
    assert valid_bhw.any()
    assert (prediction.depth_bhw[valid_bhw] > 0.0).all()
    assert (prediction.metric_scale_b > 0.0).all()
    assert (prediction.intrinsics_b33[:, 0, 0] > 0.0).all()
    torch.testing.assert_close(prediction.intrinsics_b33[:, 0, 2], torch.full((2,), 0.5, device="cuda"), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(prediction.intrinsics_b33[:, 1, 2], torch.full((2,), 0.5, device="cuda"), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(prediction.points_bhw3[..., 2], prediction.depth_bhw, atol=0.0, rtol=0.0)


@requires_cuda
@pytest.mark.parametrize("encoder", ("vitl", "vitb", "vits"))
def test_trt_matches_torch_twin_at_default_arkitscenes_resolution(encoder: Encoder) -> None:
    """TensorRT and ONNX-compatible torch geometry meet the parity bounds for every encoder."""
    from monopriors.models.monoprior.moge_v2_trt import (
        DEFAULT_IMAGE_HW,
        MoGeV2GeometryOutput,
        MoGeV2TorchGeometryPredictor,
        MoGeV2TrtGeometryPredictor,
    )

    rgb_bhw3: UInt8[Tensor, "2 756 1008 3"] = _synthetic_rgb_batch(2, DEFAULT_IMAGE_HW)
    rgb_bhw3[1] = torch.flip(rgb_bhw3[1], dims=(1,))
    trt_predictor: MoGeV2TrtGeometryPredictor = MoGeV2TrtGeometryPredictor(encoder=encoder, batch_size=8)
    torch_predictor: MoGeV2TorchGeometryPredictor = MoGeV2TorchGeometryPredictor(encoder=encoder)

    trt_prediction: MoGeV2GeometryOutput = trt_predictor(rgb_bhw3)
    torch_prediction: MoGeV2GeometryOutput = torch_predictor(rgb_bhw3)
    torch.cuda.synchronize()

    cosine_bhw: Float32[Tensor, "2 756 1008"] = torch.sum(
        trt_prediction.normals_bhw3 * torch_prediction.normals_bhw3,
        dim=-1,
    ).clamp(-1.0, 1.0)
    angular_error_bhw: Float32[Tensor, "2 756 1008"] = torch.rad2deg(torch.acos(cosine_bhw))
    mean_error_degrees: float = angular_error_bhw.mean().item()
    p99_error_degrees: float = torch.quantile(angular_error_bhw, 0.99).item()
    assert mean_error_degrees < 0.2 and p99_error_degrees < 0.5, (
        f"normal parity: mean={mean_error_degrees:.6f} degrees, p99={p99_error_degrees:.6f} degrees"
    )

    valid_bhw: Bool[Tensor, "2 756 1008"] = (trt_prediction.mask_bhw > 0.5) & (torch_prediction.mask_bhw > 0.5)
    assert valid_bhw.any()
    relative_depth_error_valid: Float32[Tensor, "valid"] = (
        (trt_prediction.depth_bhw - torch_prediction.depth_bhw).abs() / torch_prediction.depth_bhw.abs().clamp_min(1e-6)
    )[valid_bhw]
    assert torch.median(relative_depth_error_valid).item() < 0.01
    assert torch.quantile(relative_depth_error_valid, 0.95).item() < 0.03
    relative_point_error_valid: Float32[Tensor, "valid"] = (
        torch.linalg.vector_norm(trt_prediction.points_bhw3 - torch_prediction.points_bhw3, dim=-1)
        / torch.linalg.vector_norm(torch_prediction.points_bhw3, dim=-1).clamp_min(1e-6)
    )[valid_bhw]
    assert torch.median(relative_point_error_valid).item() < 0.01
    assert torch.quantile(relative_point_error_valid, 0.95).item() < 0.03
    focal_relative_error_b2: Float32[Tensor, "2 2"] = (
        (trt_prediction.intrinsics_b33[:, (0, 1), (0, 1)] - torch_prediction.intrinsics_b33[:, (0, 1), (0, 1)]).abs()
        / torch_prediction.intrinsics_b33[:, (0, 1), (0, 1)].abs()
    )
    assert focal_relative_error_b2.max().item() < 0.01
    torch.testing.assert_close(trt_prediction.metric_scale_b, torch_prediction.metric_scale_b, rtol=0.01, atol=1e-5)
    torch.testing.assert_close(trt_prediction.shift_b, torch_prediction.shift_b, rtol=0.0, atol=0.01)
    torch.testing.assert_close(trt_prediction.points_bhw3[..., 2], trt_prediction.depth_bhw, atol=0.0, rtol=0.0)

    held_prediction: MoGeV2GeometryOutput = trt_predictor(rgb_bhw3[:1])
    held_copies: tuple[Float32[Tensor, "*shape"], ...] = tuple(output_tensor.clone() for output_tensor in held_prediction)
    reused_prediction: MoGeV2GeometryOutput = trt_predictor(torch.flip(rgb_bhw3[:1], dims=(2,)))
    for held_tensor, held_copy, reused_tensor in zip(held_prediction, held_copies, reused_prediction, strict=True):
        assert reused_tensor.data_ptr() != held_tensor.data_ptr()
        torch.testing.assert_close(held_tensor, held_copy)


@requires_cuda
@pytest.mark.parametrize("encoder", ("vitl", "vitb", "vits"))
@pytest.mark.parametrize("caller_hw", ((756, 1008), (720, 1280)), ids=("4:3", "16:9"))
def test_trt_geometry_matches_vendored_infer_convention(encoder: Encoder, caller_hw: tuple[int, int]) -> None:
    """TRT matches vendored ``infer`` (raw points, metric, RDF, normalized K) within tolerance.

    Bars are set from measured errors on ``room.jpg`` (2026-08-28, RTX 5090):
    depth/points median <= 0.7 %, p95 <= 0.85 %; focal <= 0.51 %; normals
    mean <= 0.56 deg, p99 <= 4.3 deg; mask agreement 100 %. The residual is the
    resample path (``infer`` resizes once inside the encoder; the batched
    predictor resizes to the bucket and back), not the engine: the torch twin
    sits at the same distance from ``infer``.
    """
    from einops import rearrange

    from monopriors.models.monoprior.moge_v2_trt import MoGeV2GeometryOutput, MoGeV2TrtGeometryPredictor
    from monopriors.models.surface_normal.moge_v2 import MOGE_V2_NORMAL_CHECKPOINTS
    from monopriors.third_party.moge.model._inference import InferenceOutput
    from monopriors.third_party.moge.model.v2 import MoGeModel

    image_path: Path = Path("data/examples/single-image/room.jpg")
    bgr_hw3: UInt8[np.ndarray, "h0 w0 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_hw3 is None:
        pytest.skip(f"{image_path} missing; run the monoprior download task")
    resized_bgr_hw3: UInt8[np.ndarray, "h w 3"] = cv2.resize(bgr_hw3, (caller_hw[1], caller_hw[0]), interpolation=cv2.INTER_AREA)
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(resized_bgr_hw3, cv2.COLOR_BGR2RGB)
    rgb_bhw3: UInt8[Tensor, "1 h w 3"] = torch.from_numpy(rgb_hw3).cuda()[None]
    image_3hw: Float32[Tensor, "3 h w"] = rearrange(rgb_bhw3[0], "h w c -> c h w").float() / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive

    trt_predictor: MoGeV2TrtGeometryPredictor = MoGeV2TrtGeometryPredictor(encoder=encoder, batch_size=8)
    checkpoint: tuple[str, str] = MOGE_V2_NORMAL_CHECKPOINTS[encoder]
    model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
    model.onnx_compatible_mode = True

    trt_prediction: MoGeV2GeometryOutput = trt_predictor(rgb_bhw3)
    reference: InferenceOutput = model.infer(
        image_3hw,
        resolution_level=9,
        force_projection=False,
        apply_mask=False,
        output_heads=("points", "normal", "mask", "scale"),
    )
    reference_points_hw3: Float32[Tensor, "h w 3"] = reference["points"]
    reference_depth_hw: Float32[Tensor, "h w"] = reference["depth"]
    reference_intrinsics_33: Float32[Tensor, "3 3"] = reference["intrinsics"]
    reference_normals_hw3: Float32[Tensor, "h w 3"] = reference["normal"]
    reference_mask_hw: Bool[Tensor, "h w"] = reference["mask"]

    finite_hw: Bool[Tensor, "h w"] = torch.isfinite(reference_depth_hw)
    assert finite_hw.float().mean().item() > 0.5
    relative_depth_error_valid: Float32[Tensor, "valid"] = (
        (trt_prediction.depth_bhw[0] - reference_depth_hw).abs() / reference_depth_hw.abs().clamp_min(1e-6)
    )[finite_hw]
    depth_median: float = torch.median(relative_depth_error_valid).item()
    depth_p95: float = torch.quantile(relative_depth_error_valid, 0.95).item()
    assert depth_median < 0.01 and depth_p95 < 0.03, f"depth: median={depth_median:.5f} p95={depth_p95:.5f}"
    point_relative_error_valid: Float32[Tensor, "valid"] = (
        torch.linalg.vector_norm(trt_prediction.points_bhw3[0] - reference_points_hw3, dim=-1)
        / torch.linalg.vector_norm(reference_points_hw3, dim=-1).clamp_min(1e-6)
    )[finite_hw]
    points_median: float = torch.median(point_relative_error_valid).item()
    points_p95: float = torch.quantile(point_relative_error_valid, 0.95).item()
    assert points_median < 0.01 and points_p95 < 0.03, f"points: median={points_median:.5f} p95={points_p95:.5f}"
    intrinsics_indices: tuple[tuple[int, ...], tuple[int, ...]] = ((0, 1, 0, 1), (0, 1, 2, 2))
    trt_intrinsics_4: Float32[Tensor, "4"] = trt_prediction.intrinsics_b33[0][intrinsics_indices]
    reference_intrinsics_4: Float32[Tensor, "4"] = reference_intrinsics_33[intrinsics_indices]
    intrinsics_relative_error_4: Float32[Tensor, "4"] = (trt_intrinsics_4 - reference_intrinsics_4).abs() / reference_intrinsics_4.abs()
    assert intrinsics_relative_error_4.max().item() < 0.01, f"intrinsics: {intrinsics_relative_error_4.tolist()}"
    cosine_valid: Float32[Tensor, "valid"] = torch.sum(trt_prediction.normals_bhw3[0] * reference_normals_hw3, dim=-1)[finite_hw].clamp(-1.0, 1.0)
    angular_error_valid: Float32[Tensor, "valid"] = torch.rad2deg(torch.acos(cosine_valid))
    normal_mean_degrees: float = angular_error_valid.mean().item()
    normal_p99_degrees: float = torch.quantile(angular_error_valid, 0.99).item()
    assert normal_mean_degrees < 1.0 and normal_p99_degrees < 5.0, f"normals: mean={normal_mean_degrees:.4f} p99={normal_p99_degrees:.4f} deg"
    mask_agreement: float = ((trt_prediction.mask_bhw[0] > 0.5) == reference_mask_hw).float().mean().item()
    assert mask_agreement > 0.99, f"mask agreement: {mask_agreement:.5f}"


@requires_cuda
def test_bucketed_trt_predictor_handles_widescreen_caller_resolution() -> None:
    """A 16:9 caller input uses its bucket and returns caller-sized tensors."""
    from monopriors.models.monoprior.moge_v2_trt import ASPECT_BUCKETS, MoGeV2GeometryOutput, MoGeV2TrtGeometryPredictor

    caller_hw: tuple[int, int] = (180, 320)
    widescreen_bucket: tuple[int, int] = (630, 1120)
    assert widescreen_bucket in ASPECT_BUCKETS
    rgb_bhw3: UInt8[Tensor, "1 180 320 3"] = _synthetic_rgb_batch(1, caller_hw)
    predictor: MoGeV2TrtGeometryPredictor = MoGeV2TrtGeometryPredictor(batch_size=8)

    prediction: MoGeV2GeometryOutput = predictor(rgb_bhw3)

    assert set(predictor.runtimes) == {widescreen_bucket}
    assert prediction.depth_bhw.shape == (1, 180, 320)
    assert prediction.points_bhw3.shape == (1, 180, 320, 3)
    assert prediction.normals_bhw3.shape == (1, 180, 320, 3)
    assert prediction.mask_bhw.shape == (1, 180, 320)
    assert prediction.intrinsics_b33.shape == (1, 3, 3)
    assert prediction.metric_scale_b.shape == (1,)
    assert prediction.shift_b.shape == (1,)
    torch.testing.assert_close(prediction.points_bhw3[..., 2], prediction.depth_bhw, atol=0.0, rtol=0.0)
