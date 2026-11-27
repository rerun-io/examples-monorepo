"""The learned-feature runners on ONNX Runtime's CPU provider, and how far TensorRT lands from them.

Engine-free runs (a machine without NVIDIA, a test lane without engines) take `inference_runtime="onnxruntime"`. RaCo and
LightGlue+ run the very graphs the engines are built from. FlexGlue on ONNX Runtime runs COLMAP's own fp32 graphs as
shipped, the upstream reference; TensorRT runs colsfm's rewrite of them (standard-op attention, folded options, FP8 on
sm_89+). The bands below therefore measure TensorRT's precision and rewrite against fp32, and the engine-free lanes do not
exercise the rewrite itself (`test_flexglue_trt.py` does).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final, get_args

import numpy as np
import pytest
import torch
from jaxtyping import Float32, Int64
from numpy import ndarray

from colsfm.database import ImagePair, KeypointsXY, create_database, read_keypoints
from colsfm.features import ExtractionReport, FeatureOptions, extract_features
from colsfm.features_native import ExtractionGroup, ImageTask
from colsfm.features_raco import extraction_groups
from colsfm.frames_meta import FramesMeta
from colsfm.inference import InferenceRuntime
from colsfm.matching import MatchingBackend, MatchingOptions, MatchReport, match_pairs
from colsfm.model_assets import RACO_LIGHTGLUE_ONNX_PATH, required_raco_graphs
from colsfm.pairs import select_pairs

RACO_KEYPOINT_SHARE: Final[float] = 0.93
"""Smallest per-image share of TensorRT's RaCo keypoints that ONNX Runtime finds within 0.5 px. Measured 0.940-0.976 on
Galileo's first rig frame (RTX 5090, 2026-11-27): the engine runs TF32, and the top-2048 cut swaps near-ties."""
MATCH_INLIER_DIFFERENCE: Final[dict[MatchingBackend, float]] = {"flexglue": 0.03, "raco": 0.03}
"""Largest summed per-pair |ONNX Runtime - TensorRT| verified-inlier difference, as a share of TensorRT's total. Measured on
the same frame: FlexGlue 0.010 (589 = 589 inliers; TensorRT runs its FP8 matcher), LightGlue+ 0.008 (1487 vs 1499)."""


def test_onnx_runtime_groups_carry_the_graphs_and_build_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_engine(*_: object, **__: object) -> Path:
        raise AssertionError("an ONNX Runtime run must not build a TensorRT engine")

    monkeypatch.setattr("colsfm.features_raco.resolve_engine", no_engine)
    tasks: list[ImageTask] = [ImageTask(1, "a.jpeg", 1920, 1200), ImageTask(2, "b.jpeg", 800, 800)]
    groups: list[ExtractionGroup] = extraction_groups(tasks, native_resolution=True, choice="auto", inference_runtime="onnxruntime")
    assert [group.engine_path.suffix for group in groups] == [".onnx", ".onnx"]
    assert [group.maximum_batch_size for group in groups] == [1, 1]
    assert [[task.name for task in group.tasks] for group in groups] == [["a.jpeg"], ["b.jpeg"]]


def _graphs_and_cuda() -> bool:
    graphs: list[Path] = [*required_raco_graphs("raco", "flexglue", "onnxruntime"), RACO_LIGHTGLUE_ONNX_PATH]
    return torch.cuda.is_available() and all(path.is_file() for path in graphs)


@dataclass(frozen=True, slots=True)
class RuntimeRun:
    """One Galileo rig frame extracted and matched on one runtime, one database per matcher."""

    extraction: ExtractionReport
    """The one RaCo pass both matchers' databases copy."""
    databases: dict[MatchingBackend, Path]
    """The database each matcher wrote."""
    reports: dict[MatchingBackend, MatchReport]
    """What each matcher reported."""


MATCHERS: Final[tuple[MatchingBackend, ...]] = ("flexglue", "raco")
"""The two matchers with an engine: FlexGlue (the default) and LightGlue+."""


@pytest.fixture(scope="module")
def galileo_on_both_runtimes(
    galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[InferenceRuntime, RuntimeRun]:
    """The first Galileo rig frame through RaCo and both matchers, on TensorRT and on ONNX Runtime's CPU provider."""
    if not _graphs_and_cuda():
        pytest.skip("needs the RaCo / LightGlue+ / FlexGlue graphs and a CUDA device (TensorRT is the reference)")
    subset: FramesMeta = galileo_input.filtered(list(galileo_input.rig_frames()[0].keyframe_ids))
    names: list[str] = [keyframe.image_name for keyframe in subset.keyframes]
    pairs: list[ImagePair] = select_pairs(subset, connected_keyframe_num=1)
    runs: dict[InferenceRuntime, RuntimeRun] = {}
    for runtime in get_args(InferenceRuntime):
        extracted_path: Path = tmp_path_factory.mktemp(f"{runtime}-features") / "galileo.db"
        create_database(extracted_path, subset)
        extraction: ExtractionReport = extract_features(
            extracted_path, galileo_input_dir, names, FeatureOptions(backend="raco", inference_runtime=runtime)
        )
        databases: dict[MatchingBackend, Path] = {}
        reports: dict[MatchingBackend, MatchReport] = {}
        for matcher in MATCHERS:
            # One extraction per runtime: each matcher writes its own copy, since a database takes one set of matches.
            database_path: Path = tmp_path_factory.mktemp(f"{runtime}-{matcher}") / "galileo.db"
            shutil.copyfile(extracted_path, database_path)
            reports[matcher] = match_pairs(
                database_path, pairs, MatchingOptions(backend=matcher, inference_runtime=runtime, image_dir=galileo_input_dir)
            )
            databases[matcher] = database_path
        runs[runtime] = RuntimeRun(extraction, databases, reports)
    return runs


@pytest.mark.integration
def test_raco_on_onnx_runtime_finds_the_tensorrt_keypoints(galileo_on_both_runtimes: dict[InferenceRuntime, RuntimeRun]) -> None:
    tensorrt: RuntimeRun = galileo_on_both_runtimes["tensorrt"]
    onnx: RuntimeRun = galileo_on_both_runtimes["onnxruntime"]
    assert onnx.extraction.device == "cpu" and tensorrt.extraction.device == "cuda"
    assert set(onnx.extraction.keypoint_counts) == set(tensorrt.extraction.keypoint_counts)
    shares: list[float] = []
    for image_id in tensorrt.extraction.keypoint_counts:
        reference: KeypointsXY = read_keypoints(tensorrt.databases["flexglue"], image_id)
        candidate: KeypointsXY = read_keypoints(onnx.databases["flexglue"], image_id)
        distances_px: Float32[ndarray, "num_reference"] = np.min(
            np.linalg.norm(reference[:, None, :] - candidate[None, :, :], axis=2), axis=1
        ).astype(np.float32)
        shares.append(float(np.mean(distances_px <= 0.5)))
    print(
        f"[onnxruntime] RaCo: share of TensorRT keypoints with an ONNX Runtime keypoint within 0.5 px, per image: {[round(share, 4) for share in shares]}"
    )
    assert min(shares) >= RACO_KEYPOINT_SHARE


@pytest.mark.integration
@pytest.mark.parametrize("matcher", MATCHERS)
def test_the_matchers_on_onnx_runtime_verify_what_tensorrt_verifies(
    galileo_on_both_runtimes: dict[InferenceRuntime, RuntimeRun], matcher: MatchingBackend
) -> None:
    tensorrt: MatchReport = galileo_on_both_runtimes["tensorrt"].reports[matcher]
    onnx: MatchReport = galileo_on_both_runtimes["onnxruntime"].reports[matcher]
    assert onnx.device == "cpu" and set(onnx.pair_stats) == set(tensorrt.pair_stats)
    reference: Int64[ndarray, "num_pairs"] = np.asarray([tensorrt.pair_stats[pair].inlier_matches for pair in tensorrt.pair_stats], dtype=np.int64)
    candidate: Int64[ndarray, "num_pairs"] = np.asarray([onnx.pair_stats[pair].inlier_matches for pair in tensorrt.pair_stats], dtype=np.int64)
    relative: float = float(np.abs(candidate - reference).sum() / max(int(reference.sum()), 1))
    print(
        f"[onnxruntime] {matcher}: verified inliers TensorRT {int(reference.sum())}, ONNX Runtime {int(candidate.sum())}, summed |difference| {relative:.4f} of TensorRT's"
    )
    assert relative <= MATCH_INLIER_DIFFERENCE[matcher]
