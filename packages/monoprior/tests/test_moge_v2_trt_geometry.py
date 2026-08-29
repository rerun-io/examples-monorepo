"""Contract, parity, and convention tests for batched full MoGe v2 geometry."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import cv2
import numpy as np
import pytest
import torch
from conftest import requires_cuda, slow_cuda, synthetic_rgb_batch
from jaxtyping import Bool, Float32, UInt8
from torch import Tensor

from monopriors.models.moge_v2_trt_shared import Encoder
from monopriors.models.monoprior.moge_v2_trt import MoGeV2GeometryOutput


class GeometryCloseBars(NamedTuple):
    """Maximum parity errors accepted by the shared geometry assertion."""

    depth_median: float
    """Maximum median relative depth error."""
    depth_p95: float
    """Maximum 95th-percentile relative depth error."""
    points_median: float
    """Maximum median relative point error."""
    points_p95: float
    """Maximum 95th-percentile relative point error."""
    intrinsics_max: float
    """Maximum relative focal and principal-point error."""
    normal_mean_degrees: float
    """Maximum mean normal angular error in degrees."""
    normal_p99_degrees: float
    """Maximum 99th-percentile normal angular error in degrees."""
    mask_agreement: float
    """Minimum binary mask agreement fraction."""


def _assert_geometry_close(
    actual: MoGeV2GeometryOutput,
    reference_depth_bhw: Float32[Tensor, "b h w"],
    reference_points_bhw3: Float32[Tensor, "b h w 3"],
    reference_normals_bhw3: Float32[Tensor, "b h w 3"],
    reference_mask_bhw: Bool[Tensor, "b h w"] | Float32[Tensor, "b h w"],
    reference_intrinsics_b33: Float32[Tensor, "b 3 3"],
    bars: GeometryCloseBars,
) -> None:
    """Assert common depth, point, normal, mask, and intrinsics parity."""
    valid_bhw: Bool[Tensor, "b h w"] = (
        torch.isfinite(reference_depth_bhw) & (actual.mask_bhw > 0.5) & (reference_mask_bhw > 0.5)
    )
    assert valid_bhw.any()
    relative_depth_error_valid: Float32[Tensor, "valid"] = (
        (actual.depth_bhw - reference_depth_bhw).abs() / reference_depth_bhw.abs().clamp_min(1e-6)
    )[valid_bhw]
    depth_median: float = torch.median(relative_depth_error_valid).item()
    depth_p95: float = torch.quantile(relative_depth_error_valid, 0.95).item()
    assert depth_median < bars.depth_median and depth_p95 < bars.depth_p95, f"depth: median={depth_median:.5f} p95={depth_p95:.5f}"

    relative_point_error_valid: Float32[Tensor, "valid"] = (
        torch.linalg.vector_norm(actual.points_bhw3 - reference_points_bhw3, dim=-1)
        / torch.linalg.vector_norm(reference_points_bhw3, dim=-1).clamp_min(1e-6)
    )[valid_bhw]
    points_median: float = torch.median(relative_point_error_valid).item()
    points_p95: float = torch.quantile(relative_point_error_valid, 0.95).item()
    assert points_median < bars.points_median and points_p95 < bars.points_p95, f"points: median={points_median:.5f} p95={points_p95:.5f}"

    intrinsics_indices: tuple[tuple[int, ...], tuple[int, ...]] = ((0, 1, 0, 1), (0, 1, 2, 2))
    actual_intrinsics_b4: Float32[Tensor, "b 4"] = actual.intrinsics_b33[:, intrinsics_indices[0], intrinsics_indices[1]]
    reference_intrinsics_b4: Float32[Tensor, "b 4"] = reference_intrinsics_b33[:, intrinsics_indices[0], intrinsics_indices[1]]
    intrinsics_relative_error_b4: Float32[Tensor, "b 4"] = (
        (actual_intrinsics_b4 - reference_intrinsics_b4).abs() / reference_intrinsics_b4.abs()
    )
    assert intrinsics_relative_error_b4.max().item() < bars.intrinsics_max, f"intrinsics: {intrinsics_relative_error_b4.tolist()}"

    cosine_valid: Float32[Tensor, "valid"] = torch.sum(actual.normals_bhw3 * reference_normals_bhw3, dim=-1)[valid_bhw].clamp(-1.0, 1.0)
    angular_error_valid: Float32[Tensor, "valid"] = torch.rad2deg(torch.acos(cosine_valid))
    normal_mean_degrees: float = angular_error_valid.mean().item()
    normal_p99_degrees: float = torch.quantile(angular_error_valid, 0.99).item()
    assert normal_mean_degrees < bars.normal_mean_degrees and normal_p99_degrees < bars.normal_p99_degrees, (
        f"normals: mean={normal_mean_degrees:.4f} p99={normal_p99_degrees:.4f} deg"
    )
    mask_agreement: float = ((actual.mask_bhw > 0.5) == (reference_mask_bhw > 0.5)).float().mean().item()
    assert mask_agreement > bars.mask_agreement, f"mask agreement: {mask_agreement:.5f}"


def test_token_grid_hw_matches_pinned_aspect_buckets() -> None:
    """The 3,600-token budget produces the five fixed network grids."""
    from monopriors.models.moge_v2_trt_shared import ASPECT_BUCKETS, token_grid_hw

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
    from monopriors.models.moge_v2_trt_shared import ASPECT_BUCKETS, select_aspect_bucket

    selected_bucket: tuple[int, int] = select_aspect_bucket(input_hw, ASPECT_BUCKETS)

    assert selected_bucket == expected_bucket


def test_trt_predictor_builds_exact_fixed_network_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A single network option eagerly creates exactly that runtime key."""
    import monopriors.models.monoprior.moge_v2_trt as moge_v2_trt

    class FakeRuntime:
        def __init__(self, engine_path: Path, *, use_cuda_graph: bool = False) -> None:
            self.engine_path: Path = engine_path
            self.use_cuda_graph: bool = use_cuda_graph

        def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
            return inputs

    class FakeTrtBuildConfig:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def fake_ensure_engine(*_args: object, **_kwargs: object) -> Path:
        return engine_path

    engine_path: Path = tmp_path / "moge-geometry.engine"
    engine_path.write_bytes(b"engine")
    onnx_path: Path = tmp_path / "moge-geometry.onnx"
    monkeypatch.setattr(moge_v2_trt, "export_moge_v2_onnx", lambda **_kwargs: onnx_path)
    monkeypatch.setitem(
        sys.modules,
        "trtkit",
        SimpleNamespace(TrtBuildConfig=FakeTrtBuildConfig, ensure_engine=fake_ensure_engine),
    )
    monkeypatch.setitem(sys.modules, "trtkit.tensorrt_runtime", SimpleNamespace(TensorRtRuntime=FakeRuntime))

    network_hw: tuple[int, int] = (728, 966)
    predictor: moge_v2_trt.MoGeV2TrtGeometryPredictor = moge_v2_trt.MoGeV2TrtGeometryPredictor(
        network_hw_options=(network_hw,),
        use_cuda_graph=True,
    )

    assert set(predictor.runtimes) == {network_hw}
    assert predictor.runtimes[network_hw].engine_path == engine_path
    assert predictor.runtimes[network_hw].use_cuda_graph


@slow_cuda
@requires_cuda
def test_torch_predictor_returns_batched_metric_geometry() -> None:
    """The torch twin returns owning CUDA tensors in the geometry contract."""
    from monopriors.models.monoprior.moge_v2_trt import MoGeV2GeometryOutput, MoGeV2TorchGeometryPredictor

    image_hw: tuple[int, int] = (56, 70)
    rgb_bhw3: UInt8[Tensor, "2 56 70 3"] = synthetic_rgb_batch(2, image_hw)
    predictor: MoGeV2TorchGeometryPredictor = MoGeV2TorchGeometryPredictor(
        encoder="vits",
        network_hw_options=(image_hw,),
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


@slow_cuda
@requires_cuda
def test_trt_matches_torch_twin_at_caller_resolution() -> None:
    """ViT-L TensorRT and torch geometry meet the caller-resolution parity bounds."""
    from monopriors.models.monoprior.moge_v2_trt import MoGeV2TorchGeometryPredictor, MoGeV2TrtGeometryPredictor
    from monopriors.models.surface_normal.moge_v2_trt import DEFAULT_IMAGE_HW

    rgb_bhw3: UInt8[Tensor, "2 756 1008 3"] = synthetic_rgb_batch(2, DEFAULT_IMAGE_HW)
    rgb_bhw3[1] = torch.flip(rgb_bhw3[1], dims=(1,))
    trt_predictor: MoGeV2TrtGeometryPredictor = MoGeV2TrtGeometryPredictor(encoder="vitl", batch_size=8)
    torch_predictor: MoGeV2TorchGeometryPredictor = MoGeV2TorchGeometryPredictor(encoder="vitl")

    trt_prediction: MoGeV2GeometryOutput = trt_predictor(rgb_bhw3)
    torch_prediction: MoGeV2GeometryOutput = torch_predictor(rgb_bhw3)
    torch.cuda.synchronize()

    _assert_geometry_close(
        trt_prediction,
        torch_prediction.depth_bhw,
        torch_prediction.points_bhw3,
        torch_prediction.normals_bhw3,
        torch_prediction.mask_bhw,
        torch_prediction.intrinsics_b33,
        GeometryCloseBars(0.01, 0.03, 0.01, 0.03, 0.01, 0.2, 0.5, 0.99),
    )
    torch.testing.assert_close(trt_prediction.metric_scale_b, torch_prediction.metric_scale_b, rtol=0.01, atol=1e-5)
    torch.testing.assert_close(trt_prediction.shift_b, torch_prediction.shift_b, rtol=0.0, atol=0.01)
    torch.testing.assert_close(trt_prediction.points_bhw3[..., 2], trt_prediction.depth_bhw, atol=0.0, rtol=0.0)

    held_prediction: MoGeV2GeometryOutput = trt_predictor(rgb_bhw3[:1])
    held_copies: tuple[Float32[Tensor, "*shape"], ...] = tuple(output_tensor.clone() for output_tensor in held_prediction)
    reused_prediction: MoGeV2GeometryOutput = trt_predictor(torch.flip(rgb_bhw3[:1], dims=(2,)))
    for held_tensor, held_copy, reused_tensor in zip(held_prediction, held_copies, reused_prediction, strict=True):
        assert reused_tensor.data_ptr() != held_tensor.data_ptr()
        torch.testing.assert_close(held_tensor, held_copy)


@slow_cuda
@requires_cuda
@pytest.mark.parametrize(
    ("encoder", "caller_hw"),
    (("vitl", (756, 1008)), ("vitl", (720, 1280)), ("vits", (756, 1008))),
    ids=("vitl-4:3", "vitl-16:9", "vits-4:3"),
)
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

    from monopriors.models.moge_v2_trt_shared import load_pinned_moge_v2
    from monopriors.models.monoprior.moge_v2_trt import MoGeV2TrtGeometryPredictor
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
    loaded: tuple[MoGeModel, int] = load_pinned_moge_v2(encoder, 9)
    model: MoGeModel = loaded[0]

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

    _assert_geometry_close(
        trt_prediction,
        reference_depth_hw[None],
        reference_points_hw3[None],
        reference_normals_hw3[None],
        reference_mask_hw[None],
        reference_intrinsics_33[None],
        GeometryCloseBars(0.01, 0.03, 0.01, 0.03, 0.01, 1.0, 5.0, 0.99),
    )
