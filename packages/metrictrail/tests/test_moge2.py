"""Our MoGe-2 provider (metrictrail.moge2): registry hook, backend selection, token grid and the per-frame forward cache.

CPU only: a fake predictor stands in for monopriors' TensorRT / torch MoGe-2, so no GPU and no checkpoint is needed.
"""

from typing import NamedTuple

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch comes from the metrictrail env's cuda feature")
vipe_depth = pytest.importorskip("vipe.metrictrail.depth", reason="vipe comes from the metrictrail env (recipes/vipe-src)")

from vipe.utils.model_cache import ModelCache  # noqa: E402

from metrictrail import moge2 as depth  # noqa: E402


class FakeGraph(NamedTuple):
    """The fields of monopriors' MoGeV2GeometryGraphOutput the provider reads."""

    points_bhw3: "torch.Tensor"
    normals_bhw3: "torch.Tensor"
    mask_bhw: "torch.Tensor"
    metric_scale_b: "torch.Tensor"


class FakePredictor:
    """A planar affine point map that depends on the image, at a fixed network size; counts its forwards."""

    def __init__(self, network_hw: tuple[int, int]) -> None:
        self.network_hw: tuple[int, int] = network_hw
        self.calls: int = 0

    def forward_geometry(self, rgb_u8_bhw3: "torch.Tensor") -> FakeGraph:
        self.calls += 1
        height, width = self.network_hw
        brightness: float = float(rgb_u8_bhw3.float().mean()) / 255.0
        v, u = torch.meshgrid(torch.linspace(-0.5, 0.5, height), torch.linspace(-0.6, 0.6, width), indexing="ij")
        z: torch.Tensor = 1.0 + 0.2 * v + 0.1 * brightness
        points: torch.Tensor = torch.stack([u * (z + 0.3), v * (z + 0.3), z], dim=-1)[None]
        normals: torch.Tensor = torch.nn.functional.normalize(torch.stack([0.1 * u, -torch.ones_like(u), 0.2 * v], dim=-1), dim=-1)[None]
        mask: torch.Tensor = torch.full((1, height, width), 0.9)
        mask[:, :2] = 0.1
        return FakeGraph(points_bhw3=points, normals_bhw3=normals, mask_bhw=mask, metric_scale_b=torch.tensor([2.5]))


def make_engine(cache_frames: int = 8) -> tuple["depth.MoGe2Engine", dict[tuple[int, int], FakePredictor]]:
    built: dict[tuple[int, int], FakePredictor] = {}

    def factory(network_hw: tuple[int, int]) -> FakePredictor:
        built[network_hw] = FakePredictor(network_hw)
        return built[network_hw]

    return depth.MoGe2Engine("tensorrt", cache_frames=cache_frames, predictor_factory=factory), built


def frame(seed: int, height: int = 30, width: int = 40) -> "torch.Tensor":
    """A (1,3,H,W) float RGB batch holding 8-bit values, like MetricTrail's canonical frames."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, 3, height, width), generator=generator).float() / 255.0


@pytest.mark.parametrize(
    ("choice", "available", "expected"),
    [("auto", True, "tensorrt"), ("auto", False, "torch"), ("torch", True, "torch"), ("tensorrt", True, "tensorrt")],
)
def test_select_backend(choice: str, available: bool, expected: str) -> None:
    assert depth.select_moge2_backend(choice, tensorrt_available=available) == expected


def test_select_backend_refuses_tensorrt_without_tensorrt_and_unknown_names() -> None:
    with pytest.raises(RuntimeError, match="tensorrt"):
        depth.select_moge2_backend("tensorrt", tensorrt_available=False)
    with pytest.raises(ValueError, match="unknown"):
        depth.select_moge2_backend("onnx", tensorrt_available=True)


def test_provider_reads_the_backend_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(depth.MOGE2_BACKEND_ENV, "torch")
    monkeypatch.setattr(depth, "tensorrt_importable", lambda: True)
    provider = depth.MoGe2NormalProvider("l")  # builds nothing until the first forward
    assert provider.engine.backend == "torch"
    assert provider.precision == "FP16-autocast"
    monkeypatch.setenv(depth.MOGE2_BACKEND_ENV, "auto")
    tensorrt = depth.MoGe2NormalProvider("l")
    assert tensorrt.engine.backend == "tensorrt"
    assert tensorrt.precision == "TensorRT-FP16"


def test_provider_refuses_fp32_instead_of_ignoring_it() -> None:
    with pytest.raises(ValueError, match="FP16 on every backend"):
        depth.MoGe2NormalProvider("l", use_fp16=False)


def test_network_size_is_the_inputs_own_token_grid_not_a_bucket() -> None:
    assert depth.MoGe2Engine.network_hw((388, 518)) == (728, 966)
    kitti_hw: tuple[int, int] = depth.MoGe2Engine.network_hw((370, 1226))
    assert kitti_hw == (462, 1526)  # KITTI keeps its ~3.3:1; the nearest (16:9) bucket would squash it to (630, 1120)
    assert kitti_hw[0] % 14 == 0 and kitti_hw[1] % 14 == 0


def test_resolution_level_selects_the_grid_and_separates_cached_models() -> None:
    assert depth.MoGe2Engine.network_hw((280, 504), resolution_level=3) == (462, 840)
    cache = ModelCache()
    coarse = depth.make_provider("l", resolution_level=3, model_cache=cache)
    repeated = depth.make_provider("l", resolution_level=3, model_cache=cache)
    original = depth.make_provider("l", resolution_level=9, model_cache=cache)
    assert coarse.engine is repeated.engine
    assert coarse.engine is not original.engine
    assert coarse.describe()["resolution_level"] == 3
    assert original.describe()["resolution_level"] == 9


@pytest.mark.parametrize("level", [-1, 10])
def test_resolution_level_rejects_out_of_range_values(level: int) -> None:
    with pytest.raises(ValueError, match="resolution_level"):
        depth.MoGe2Engine("tensorrt", resolution_level=level)


def test_a_focal_change_reuses_the_forward_and_matches_an_uncached_run() -> None:
    cached, built = make_engine()
    provider = depth.MoGe2NormalProvider("l", engine=cached)
    image: torch.Tensor = frame(0)
    first = provider.predict_metric(image, focal_px=30.0)
    second = provider.predict_metric(image, focal_px=45.0)
    again = provider.predict_metric(image, focal_px=30.0)
    assert cached.forward_count == 1 and cached.postprocess_count == 3
    assert sum(predictor.calls for predictor in built.values()) == 1
    assert not torch.equal(first.depth, second.depth)  # the focal changes the recovered shift
    torch.testing.assert_close(again.depth, first.depth, rtol=0.0, atol=0.0)

    uncached, _ = make_engine(cache_frames=0)
    fresh = depth.MoGe2NormalProvider("l", engine=uncached).predict_metric(image, focal_px=45.0)
    assert uncached.forward_count == 1
    torch.testing.assert_close(fresh.depth, second.depth, rtol=0.0, atol=0.0)
    torch.testing.assert_close(fresh.valid, second.valid, rtol=0.0, atol=0.0)
    assert fresh.normals is not None and second.normals is not None
    torch.testing.assert_close(fresh.normals, second.normals, rtol=0.0, atol=0.0)


def test_outputs_are_on_the_input_raster_and_follow_the_validity_rule() -> None:
    engine, _ = make_engine()
    prediction = depth.MoGe2NormalProvider("l", engine=engine).predict_metric(frame(1), focal_px=35.0)
    assert prediction.depth.shape == (1, 30, 40)
    assert prediction.normals is not None and prediction.normals.shape == (1, 30, 40, 3)
    torch.testing.assert_close(prediction.normals.norm(dim=-1), torch.ones(1, 30, 40))
    assert not bool(prediction.valid[0, 0].any())  # the fake's low-probability top rows
    valid_depth: torch.Tensor = prediction.depth[prediction.valid]
    assert bool((valid_depth >= depth.METRIC_DEPTH_MIN_M).all()) and bool((valid_depth <= depth.METRIC_DEPTH_MAX_M).all())


def test_cache_is_keyed_by_content_and_bounded() -> None:
    engine, _ = make_engine(cache_frames=2)
    provider = depth.MoGe2NormalProvider("l", engine=engine)
    for seed in (0, 1, 0, 2, 0, 1):
        provider.predict_metric(frame(seed), focal_px=40.0)
    # 0 (miss), 1 (miss), 0 (hit), 2 (miss, evicts 1), 0 (hit), 1 (miss: evicted)
    assert engine.forward_count == 4
    assert engine.postprocess_count == 6
    assert engine.describe()["forward_count"] == 4


def test_free_fov_focal_reuses_the_cached_forward() -> None:
    engine, _ = make_engine()
    provider = depth.MoGe2NormalProvider("l", engine=engine)
    image: torch.Tensor = frame(3)
    k4: np.ndarray = provider.estimate_free_fov_k4(image)
    provider.predict_metric(image, focal_px=float(k4[0]))
    assert engine.forward_count == 1
    assert k4.shape == (4,) and np.isfinite(k4).all() and k4[0] > 0
    np.testing.assert_allclose(k4[2:], [20.0, 15.0])


def test_registered_name_builds_our_provider_with_one_engine_per_cache() -> None:
    depth.register_providers()
    assert set(depth.PROVIDER_NAMES) <= set(vipe_depth.DEPTH_PROVIDERS)
    cache = ModelCache()
    calibration = vipe_depth.make_metrictrail_depth_model(depth.DEFAULT_PROVIDER, use_fp16=True, model_cache=cache)
    tracking = vipe_depth.make_metrictrail_depth_model(depth.DEFAULT_PROVIDER, use_fp16=True, model_cache=cache)
    assert isinstance(calibration, depth.MoGe2NormalProvider) and isinstance(tracking, vipe_depth.MetricTrailMoGe2)
    assert calibration.engine is tracking.engine  # one forward cache across calibration and tracking
    assert tracking.describe()["registry_name"] == depth.DEFAULT_PROVIDER
    assert f"pipeline.init.calibration.depth_model={depth.DEFAULT_PROVIDER}" in depth.PROVIDER_OVERRIDES
    assert f"pipeline.slam.keyframe_depth={depth.DEFAULT_PROVIDER}" in depth.PROVIDER_OVERRIDES
    # The calibration config states the precision our provider runs (upstream's default is FP32 for ViPE's moge2-l).
    assert "pipeline.init.calibration.depth_use_fp16=true" in depth.PROVIDER_OVERRIDES


def test_a_finished_run_releases_the_cached_forwards_but_keeps_the_network() -> None:
    """The app keeps its engines for its whole life; their per-frame outputs (GPU tensors) must not outlive a run."""
    engine, built = make_engine()
    provider = depth.MoGe2NormalProvider("l", engine=engine)
    provider.predict_metric(frame(0), focal_px=40.0)
    depth.clear_forward_caches()
    provider.predict_metric(frame(0), focal_px=40.0)
    assert engine.forward_count == 2  # a second forward: the first one's output was released
    assert len(built) == 1  # the same network (predictor) served both
