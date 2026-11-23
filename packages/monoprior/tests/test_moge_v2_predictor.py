"""Contracts and parity tests for the unified batched MoGe v2 predictors."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import cv2
import numpy as np
import pytest
import torch
from conftest import integration, requires_cuda, synthetic_rgb_batch
from jaxtyping import Bool, Float32, UInt8
from torch import Tensor

from monopriors.models.moge_v2 import Encoder, MoGeV2EngineProfile, MoGeV2GeometryOutput


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
    from monopriors.models.moge_v2 import ASPECT_BUCKETS, token_grid_hw

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
    from monopriors.models.moge_v2 import ASPECT_BUCKETS, select_aspect_bucket

    selected_bucket: tuple[int, int] = select_aspect_bucket(input_hw, ASPECT_BUCKETS)

    assert selected_bucket == expected_bucket


def test_dynamic_engine_serves_aspects_one_to_two_through_two_to_one() -> None:
    """``serves`` names the sizes the one dynamic engine runs (no build); a static engine serves every size."""
    from monopriors.models.moge_v2 import MoGeV2TrtPredictor

    dynamic: MoGeV2TrtPredictor = MoGeV2TrtPredictor(encoder="vitl", heads="geometry")
    assert dynamic.serves((1028, 752)) and dynamic.serves((1280, 704))  # MAMMA landscape-ish, 4DAnyone 9:16
    assert not dynamic.serves((278, 640))  # a 2.3:1 person crop
    assert MoGeV2TrtPredictor(encoder="vitl", heads="geometry", profile="static").serves((278, 640))


@pytest.mark.parametrize("profile", ("static", "dynamic"))
def test_trt_predictor_builds_exact_fixed_network_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: MoGeV2EngineProfile) -> None:
    """A single network option eagerly creates its runtime: keyed by that size (static) or the one engine (dynamic)."""
    import monopriors.models.moge_v2.predictor as moge_v2_predictor

    class FakeRuntime:
        def __init__(self, engine_path: Path, *, use_cuda_graph: bool = False, **kwargs: bool) -> None:
            self.engine_path: Path = engine_path
            self.use_cuda_graph: bool = use_cuda_graph
            self.kwargs: dict[str, bool] = kwargs

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
    monkeypatch.setattr(moge_v2_predictor, "export_moge_v2_onnx", lambda **_kwargs: onnx_path)
    monkeypatch.setitem(
        sys.modules,
        "trtkit",
        SimpleNamespace(
            TrtBuildConfig=FakeTrtBuildConfig,
            InputShapeProfile=FakeTrtBuildConfig,
            TensorRtDynamicRuntime=FakeRuntime,
            ensure_engine=fake_ensure_engine,
        ),
    )
    monkeypatch.setitem(sys.modules, "trtkit.tensorrt_runtime", SimpleNamespace(TensorRtRuntime=FakeRuntime))

    network_hw: tuple[int, int] = (728, 966)
    predictor: moge_v2_predictor.MoGeV2TrtPredictor = moge_v2_predictor.MoGeV2TrtPredictor(
        heads="geometry",
        network_hw_options=(network_hw,),
        use_cuda_graph=True,
        profile=profile,
    )

    key: tuple[int, int] | None = network_hw if profile == "static" else None
    assert set(predictor.runtimes) == {key}
    assert predictor.runtimes[key].engine_path == engine_path
    assert predictor.runtimes[key].use_cuda_graph
    # CUDA graphs replay fixed addresses: the dynamic runtime then copies into max-shape buffers and memory.
    expected_kwargs: dict[str, bool] = {} if profile == "static" else {"direct_binding": False, "per_shape_activation_memory": False}
    assert predictor.runtimes[key].kwargs == expected_kwargs


def test_torch_typed_entry_points_reject_wrong_heads() -> None:
    """Each Torch entry point rejects a predictor built for the other head set."""
    from monopriors.models.moge_v2 import MoGeV2TorchPredictor

    rgb_bhw3: UInt8[Tensor, "1 2 3 3"] = torch.zeros((1, 2, 3, 3), dtype=torch.uint8)
    predictor: MoGeV2TorchPredictor = MoGeV2TorchPredictor.__new__(MoGeV2TorchPredictor)
    predictor.heads = "geometry"
    with pytest.raises(ValueError, match="heads='geometry'; it cannot produce 'normal'"):
        predictor.predict_normals(rgb_bhw3)
    predictor.heads = "normal"
    with pytest.raises(ValueError, match="heads='normal'; it cannot produce 'geometry'"):
        predictor.predict_geometry(rgb_bhw3)


def test_trt_typed_entry_points_reject_wrong_heads() -> None:
    """Each TensorRT entry point rejects a predictor built for the other head set."""
    from monopriors.models.moge_v2 import MoGeV2TrtPredictor

    rgb_bhw3: UInt8[Tensor, "1 2 3 3"] = torch.zeros((1, 2, 3, 3), dtype=torch.uint8)
    predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor.__new__(MoGeV2TrtPredictor)
    predictor.heads = "geometry"
    with pytest.raises(ValueError, match="heads='geometry'; it cannot produce 'normal'"):
        predictor.predict_normals(rgb_bhw3)
    predictor.heads = "normal"
    with pytest.raises(ValueError, match="heads='normal'; it cannot produce 'geometry'"):
        predictor.predict_geometry(rgb_bhw3)


@integration
@requires_cuda
def test_torch_predictor_returns_batched_metric_geometry() -> None:
    """The torch twin returns owning CUDA tensors in the geometry contract."""
    from monopriors.models.moge_v2 import MoGeV2GeometryOutput, MoGeV2TorchPredictor

    image_hw: tuple[int, int] = (56, 70)
    rgb_bhw3: UInt8[Tensor, "2 56 70 3"] = synthetic_rgb_batch(2, image_hw)
    predictor: MoGeV2TorchPredictor = MoGeV2TorchPredictor(
        encoder="vits",
        heads="geometry",
        network_hw_options=(image_hw,),
        resolution_level=0,
    )

    prediction: MoGeV2GeometryOutput = predictor.predict_geometry(rgb_bhw3)

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


@integration
@requires_cuda
@pytest.mark.parametrize(
    ("profile", "caller_hw"),
    (("dynamic", (756, 1008)), ("dynamic", (1280, 704)), ("static", (756, 1008))),
    ids=("dynamic-4:3", "dynamic-portrait", "static-4:3"),
)
def test_trt_matches_torch_twin_at_caller_resolution(profile: MoGeV2EngineProfile, caller_hw: tuple[int, int]) -> None:
    """ViT-L TensorRT and torch geometry meet the caller-resolution parity bounds; the portrait case is the 4DAnyone views."""
    from monopriors.models.moge_v2 import ASPECT_BUCKETS, MoGeV2TorchPredictor, MoGeV2TrtPredictor

    rgb_bhw3: UInt8[Tensor, "2 h w 3"] = synthetic_rgb_batch(2, caller_hw)
    rgb_bhw3[1] = torch.flip(rgb_bhw3[1], dims=(1,))
    trt_predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor(encoder="vitl", heads="geometry", batch_size=8, profile=profile)
    # The twin runs at the same network size: the image's own token grid (dynamic) or its aspect bucket (static).
    torch_predictor: MoGeV2TorchPredictor = MoGeV2TorchPredictor(
        encoder="vitl", heads="geometry", network_hw_options=ASPECT_BUCKETS if profile == "static" else None
    )

    trt_prediction: MoGeV2GeometryOutput = trt_predictor.predict_geometry(rgb_bhw3)
    torch_prediction: MoGeV2GeometryOutput = torch_predictor.predict_geometry(rgb_bhw3)
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
    assert trt_predictor.has_engine(caller_hw)

    held_prediction: MoGeV2GeometryOutput = trt_predictor.predict_geometry(rgb_bhw3[:1])
    held_copies: tuple[Float32[Tensor, "*shape"], ...] = tuple(output_tensor.clone() for output_tensor in held_prediction)
    reused_prediction: MoGeV2GeometryOutput = trt_predictor.predict_geometry(torch.flip(rgb_bhw3[:1], dims=(2,)))
    for held_tensor, held_copy, reused_tensor in zip(held_prediction, held_copies, reused_prediction, strict=True):
        assert reused_tensor.data_ptr() != held_tensor.data_ptr()
        torch.testing.assert_close(held_tensor, held_copy)


@pytest.mark.golden
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

    from monopriors.models.moge_v2 import MoGeV2TrtPredictor, load_pinned_moge_v2
    from monopriors.third_party.moge.model._inference import InferenceOutput
    from monopriors.third_party.moge.model.v2 import MoGeModel

    image_path: Path = Path("data/examples/single-image/room.jpg")
    bgr_hw3: UInt8[np.ndarray, "h0 w0 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_hw3 is None:
        pytest.skip(f"{image_path} missing; run the monoprior download task")
    resized_bgr_hw3: UInt8[np.ndarray, "h w 3"] = cv2.resize(bgr_hw3, (caller_hw[1], caller_hw[0]), interpolation=cv2.INTER_AREA)
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(resized_bgr_hw3, cv2.COLOR_BGR2RGB)
    rgb_bhw3: UInt8[Tensor, "1 h w 3"] = torch.from_numpy(rgb_hw3).cuda()[None]
    image_3hw: Float32[Tensor, "3 h w"] = rearrange(rgb_bhw3[0], "h w c -> c h w").float() / 255.0

    trt_predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor(encoder=encoder, heads="geometry", batch_size=8)
    loaded: tuple[MoGeModel, int] = load_pinned_moge_v2(encoder, 9)
    model: MoGeModel = loaded[0]

    trt_prediction: MoGeV2GeometryOutput = trt_predictor.predict_geometry(rgb_bhw3)
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


@integration
@requires_cuda
def test_torch_predictor_returns_batched_unit_normals_and_masks() -> None:
    """The unified Torch core honors the normal-only batch and output contract."""
    from monopriors.models.moge_v2 import MoGeV2NormalOutput, MoGeV2TorchPredictor

    image_hw: tuple[int, int] = (56, 70)
    rgb_bhw3: UInt8[Tensor, "2 56 70 3"] = synthetic_rgb_batch(2, image_hw)
    predictor: MoGeV2TorchPredictor = MoGeV2TorchPredictor(
        encoder="vits",
        heads="normal",
        network_hw_options=(image_hw,),
        resolution_level=0,
    )

    prediction: MoGeV2NormalOutput = predictor.predict_normals(rgb_bhw3)

    assert prediction.normals_bhw3.shape == (2, 56, 70, 3)
    assert prediction.normals_bhw3.dtype == torch.float32
    assert prediction.normals_bhw3.is_cuda
    assert prediction.mask_bhw.shape == (2, 56, 70)
    assert prediction.mask_bhw.dtype == torch.float32
    assert prediction.mask_bhw.is_cuda
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction.normals_bhw3, dim=-1),
        torch.ones((2, 56, 70), device="cuda"),
        atol=2e-3,
        rtol=0.0,
    )
    assert prediction.mask_bhw.min().item() >= 0.0
    assert prediction.mask_bhw.max().item() <= 1.0


@integration
@requires_cuda
def test_trt_normal_heads_match_torch_twin_and_return_owning_outputs() -> None:
    """The cached normal-only engine matches Torch and returns owning tensors."""
    from monopriors.models.moge_v2 import (
        DEFAULT_IMAGE_HW,
        MoGeV2NormalOutput,
        MoGeV2TorchPredictor,
        MoGeV2TrtPredictor,
    )

    rgb_bhw3: UInt8[Tensor, "2 756 1008 3"] = synthetic_rgb_batch(2, DEFAULT_IMAGE_HW)
    rgb_bhw3[1] = torch.flip(rgb_bhw3[1], dims=(1,))
    trt_predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor(
        heads="normal",
        network_hw_options=(DEFAULT_IMAGE_HW,),
        batch_size=8,
    )
    # The dynamic engine runs the option's token grid (728x966), which the twin derives for this aspect.
    torch_predictor: MoGeV2TorchPredictor = MoGeV2TorchPredictor(heads="normal", network_hw_options=None)

    trt_prediction: MoGeV2NormalOutput = trt_predictor.predict_normals(rgb_bhw3)
    torch_prediction: MoGeV2NormalOutput = torch_predictor.predict_normals(rgb_bhw3)
    torch.cuda.synchronize()

    cosine_bhw: Float32[Tensor, "2 756 1008"] = torch.sum(
        trt_prediction.normals_bhw3 * torch_prediction.normals_bhw3,
        dim=-1,
    ).clamp(-1.0, 1.0)
    angular_error_bhw: Float32[Tensor, "2 756 1008"] = torch.rad2deg(torch.acos(cosine_bhw))
    # The dynamic engine's fp16 tactics sit 0.15 deg (mean) from the twin, the static 728x966 engine ~0.1; the
    # geometry twin test's bar.
    assert angular_error_bhw.mean().item() < 0.2
    assert torch.quantile(angular_error_bhw, 0.99).item() < 0.5

    held_prediction: MoGeV2NormalOutput = trt_predictor.predict_normals(rgb_bhw3[:1])
    held_normals_copy_bhw3: Float32[Tensor, "1 756 1008 3"] = held_prediction.normals_bhw3.clone()
    reused_prediction: MoGeV2NormalOutput = trt_predictor.predict_normals(torch.flip(rgb_bhw3[:1], dims=(2,)))
    assert reused_prediction.normals_bhw3.data_ptr() != held_prediction.normals_bhw3.data_ptr()
    torch.testing.assert_close(held_prediction.normals_bhw3, held_normals_copy_bhw3)


def test_onnx_path_names_the_head_set_size_and_detail_level(tmp_path: Path) -> None:
    from monopriors.models.moge_v2 import moge_v2_onnx_path

    path: Path = moge_v2_onnx_path("geometry", "vitl", (364, 644), resolution_level=9, cache_dir=tmp_path)
    assert path.parent == tmp_path / "onnx"
    assert path.name.startswith("moge-v2-vitl-geometry_364x644_t") and path.suffix == ".onnx"


def test_has_engine_checks_the_caches_only(tmp_path: Path) -> None:
    """An empty cache has no engine: the check neither exports nor builds (a caller then uses the torch predictor)."""
    from monopriors.models.moge_v2 import MoGeV2TrtPredictor

    predictor = MoGeV2TrtPredictor(encoder="vitl", heads="geometry", cache_dir=tmp_path)
    assert predictor.has_engine((1280, 704)) is False
    assert not any(tmp_path.rglob("*"))


def test_dynamic_patch_range_holds_the_callers_token_grids() -> None:
    """At detail level 9 the dynamic engine spans 588-1190 px per side: every caller's grid is inside, a 3:1 pano is not."""
    from monopriors.models.moge_v2 import dynamic_patch_range, resolve_num_tokens, token_grid_hw

    num_tokens: int = resolve_num_tokens(9)
    low, high = dynamic_patch_range(num_tokens)
    assert (14 * low, 14 * high) == (588, 1190)
    # 4DAnyone portrait 704x1280 and its landscape twin, 4:3 (ARKitScenes, zipdepth, gauss-surf), the Insta360 cube
    # faces (1:1), 16:9 and 9:16 phones, the 1:2 and 2:1 ends.
    for input_hw in ((1280, 704), (704, 1280), (756, 1008), (480, 640), (1600, 1600), (1080, 1920), (1920, 1080), (1000, 500), (500, 1000)):
        grid_hw: tuple[int, int] = token_grid_hw(input_hw[1] / input_hw[0], num_tokens)
        assert low <= grid_hw[0] // 14 <= high and low <= grid_hw[1] // 14 <= high, input_hw
    wide_hw: tuple[int, int] = token_grid_hw(3.0, num_tokens)
    assert not low <= wide_hw[1] // 14 <= high


def test_dynamic_network_size_is_the_token_grid_of_the_chosen_aspect(tmp_path: Path) -> None:
    """Dynamic: an image runs at its own token grid, or at the grid of the option nearest its aspect; outside the range
    it is rejected and ``has_engine`` is false. Static keeps the option itself."""
    from monopriors.models.moge_v2 import MoGeV2TrtPredictor

    dynamic: MoGeV2TrtPredictor = MoGeV2TrtPredictor(heads="geometry", cache_dir=tmp_path)
    assert dynamic.network_hw((1280, 704)) == (1134, 616)
    assert dynamic.network_hw((704, 1280)) == (616, 1134)
    assert dynamic.network_hw((1600, 1600)) == (840, 840)
    with pytest.raises(ValueError, match="outside the dynamic engine"):
        dynamic.network_hw((400, 1400))
    assert dynamic.has_engine((400, 1400)) is False
    assert dynamic.runtimes == {}
    static: MoGeV2TrtPredictor = MoGeV2TrtPredictor(heads="geometry", cache_dir=tmp_path, profile="static")
    assert static.network_hw((1280, 704)) == (1120, 630)  # nearest of the five aspect buckets


def test_dynamic_engine_profile_spans_the_range_and_resizes_activations(tmp_path: Path) -> None:
    """One profile: batch 1..batch_size, sides 588..1190 px, tuned at batch_size x the 4:3 grid; per-shape activation memory."""
    from monopriors.models.moge_v2 import MoGeV2TrtPredictor

    predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor(heads="geometry", cache_dir=tmp_path, batch_size=6)
    onnx_path, config = predictor._engine_spec((1134, 616))
    assert onnx_path.name.startswith("moge-v2-vitl-geometry_dyn588-1190_t3600_")
    assert predictor._engine_spec((728, 966))[0] == onnx_path  # every grid shares the one graph
    assert config.runtime_activation_resize
    (image_profile,) = config.shape_profiles
    assert image_profile.name == "image"
    assert image_profile.min_shape == (1, 3, 588, 588)
    assert image_profile.opt_shape == (6, 3, 728, 966)
    assert image_profile.max_shape == (6, 3, 1190, 1190)
    static_onnx, static_config = MoGeV2TrtPredictor(heads="geometry", cache_dir=tmp_path, profile="static")._engine_spec((728, 966))
    assert static_onnx.name.startswith("moge-v2-vitl-geometry_728x966_") and static_config.shape_profiles == ()


@pytest.mark.parametrize("grid_hw", ((52, 69), (81, 44), (42, 85), (60, 60), (1, 7)))
def test_dynamic_pos_encoding_matches_dinov2_bicubic_resize(grid_hw: tuple[int, int]) -> None:
    """The traceable positional resize equals DINOv2's ``F.interpolate(scale_factor=(g + 0.1) / 37, mode="bicubic")``."""
    import torch.nn.functional as F

    from monopriors.models.moge_v2.export import dynamic_pos_encoding

    generator: torch.Generator = torch.Generator().manual_seed(0)
    pos_embed_1nc: Float32[Tensor, "1 n c"] = torch.randn((1, 1 + 37 * 37, 8), generator=generator)
    expected_1chw: Float32[Tensor, "1 c h w"] = F.interpolate(
        pos_embed_1nc[:, 1:].reshape(1, 37, 37, 8).permute(0, 3, 1, 2),
        scale_factor=((grid_hw[0] + 0.1) / 37, (grid_hw[1] + 0.1) / 37),
        mode="bicubic",
        antialias=False,
    )
    expected_1nc: Float32[Tensor, "1 n c"] = torch.cat([pos_embed_1nc[:, :1], expected_1chw.permute(0, 2, 3, 1).flatten(1, 2)], dim=1)
    # Float32 source coordinates (PyTorch computes the scale in double first): ~1e-5 of unit-scale values.
    torch.testing.assert_close(dynamic_pos_encoding(pos_embed_1nc, grid_hw, 0.1), expected_1nc, atol=1e-4, rtol=1e-4)


def test_view_plane_uv_tensor_aspect_matches_linspace() -> None:
    """The ``arange`` path (a 0-d tensor aspect, as the dynamic export passes) equals the ``linspace`` path."""
    from monopriors.third_party.moge.utils.geometry_torch import normalized_view_plane_uv

    for width, height in ((69, 52), (44, 81), (7, 3)):
        expected_hw2: Float32[Tensor, "h w 2"] = normalized_view_plane_uv(width, height, aspect_ratio=width / height, dtype=torch.float32)
        tensor_aspect: Float32[Tensor, ""] = torch.tensor(width / height, dtype=torch.float32)
        actual_hw2: Float32[Tensor, "h w 2"] = normalized_view_plane_uv(width, height, aspect_ratio=tensor_aspect, dtype=torch.float32)
        torch.testing.assert_close(actual_hw2, expected_hw2, atol=1e-6, rtol=1e-6)
