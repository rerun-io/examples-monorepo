"""Behavioral tests for the ARKitScenes PromptDA API."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest
import rerun as rr
import torch
from numpy.testing import assert_allclose, assert_array_equal
from rerun.catalog import DatasetEntry
from rerun.experimental import RrdReader
from simplecv.ops.tsdf_depth_fuser import Open3DFuser, log_fused_mesh

pytest.importorskip("pyarrow", reason="ARKitScenes catalog deps live in the PromptDA catalog lanes")
pytest.importorskip("arkitscenes_download", reason="ARKitScenes catalog deps live in the PromptDA catalog lanes")
_dataloader = pytest.importorskip("rerun.experimental.dataloader")
if not hasattr(_dataloader, "NoShuffle"):
    pytest.skip("NVDEC tests need the prerelease Rerun dataloader", allow_module_level=True)

from arkitscenes_download.ingest.paths import CONFIDENCE, DEPTH_PROMPTDA, PROMPTDA_MESH, VIDEO_WIDE  # noqa: E402
from rerun.experimental.dataloader import RerunIterableDataset  # noqa: E402
from simplecv.rerun_dataloader import SegmentNvdecDecoder  # noqa: E402

from rerun_prompt_da.apis.arkitscenes_shared import (  # noqa: E402
    filter_depth_for_fusion,
    segments_to_process,
    stride_for,
    world_t_cam_from_pose,
)
from rerun_prompt_da.apis.prompt_da_arkitscenes import (  # noqa: E402
    CompletedPromptDABatch,
    PDAArkitScenesConfig,
    fuse_and_log_batch,
    log_promptda_frame,
)
from rerun_prompt_da.promptda_stream import PromptDACollate, promptda_dataset  # noqa: E402


def test_promptda_layer_rrd_keeps_depth_and_mesh_contract(tmp_path: Path) -> None:
    """Write the registered layer's two entities with only depth on video_time."""
    rrd_path: Path = tmp_path / "promptda.rrd"
    with rr.RecordingStream("arkitscenes", recording_id="segment", send_properties=False) as recording:
        recording.save(rrd_path)
        log_promptda_frame(recording, 123_456_789, np.full((2, 3), 1500, dtype=np.uint16))
        mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32)),
        )
        mesh.compute_vertex_normals()
        log_fused_mesh(PROMPTDA_MESH, mesh, recording=recording)

    reader: RrdReader = RrdReader(rrd_path)
    chunks = list(reader.stream(store=reader.recordings()[0]).to_chunks())
    chunks_by_path = {str(chunk.entity_path): chunk for chunk in chunks}
    assert set(chunks_by_path) == {f"/{DEPTH_PROMPTDA}", f"/{PROMPTDA_MESH}"}
    assert "video_time" in chunks_by_path[f"/{DEPTH_PROMPTDA}"].to_record_batch().schema.names


def test_promptda_dataset_uses_nvdec_and_fetches_fusion_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build one natural-order NVDEC query with the confidence needed by TSDF fusion."""
    dataset = object.__new__(DatasetEntry)
    captured: dict[str, object] = {}

    def initialize_decoder(_decoder: SegmentNvdecDecoder, *args: object) -> None:
        captured["decoder_args"] = args

    def initialize_samples(_samples: RerunIterableDataset, *args: object, **kwargs: object) -> None:
        captured["dataset_args"] = args
        captured["dataset_kwargs"] = kwargs

    monkeypatch.setattr(SegmentNvdecDecoder, "__init__", initialize_decoder)
    monkeypatch.setattr(RerunIterableDataset, "__init__", initialize_samples)

    samples, decoder = promptda_dataset(dataset, "segment", 10.0, torch.device("cuda"))

    assert isinstance(samples, RerunIterableDataset)
    assert isinstance(decoder, SegmentNvdecDecoder)
    assert captured["decoder_args"] == (dataset, VIDEO_WIDE, "video_time", torch.device("cuda"), 60)
    fields = captured["dataset_kwargs"]["fields"]  # type: ignore[index]
    assert fields["video"].decode is decoder  # type: ignore[index]
    assert fields["conf"].path == f"/{CONFIDENCE}:SegmentationImage:buffer"  # type: ignore[index]
    assert captured["dataset_kwargs"]["fetch_block_size"] == 1024  # type: ignore[index]


def test_promptda_collate_keeps_stored_confidence_and_honors_ingest_rotation() -> None:
    """Prepare landscape model inputs while retaining confidence in catalog orientation."""
    samples: RerunIterableDataset = object.__new__(RerunIterableDataset)
    samples._sample_index = SimpleNamespace(  # pyrefly: ignore  # bad-assignment — minimal synthetic sampling grid
        segments=[SimpleNamespace(index_start=100)], ns_per_sample=10
    )
    collate: PromptDACollate = PromptDACollate(samples, torch.device("cpu"), quarter_turns=3, timestamp_step_ns=12)
    batch = collate([
        {
            "video": torch.arange(18, dtype=torch.uint8).reshape(3, 3, 2),
            "depth": torch.zeros((1, 256, 192), dtype=torch.uint16),
            "conf": torch.arange(256 * 192).to(torch.uint8),
            "k": torch.eye(3, dtype=torch.float32).T.reshape(9),
            "pose_t": torch.tensor([1.0, 2.0, 3.0]),
            "pose_q": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        }
    ])

    assert batch is not None
    assert batch.quarter_turns == 3
    assert tuple(batch.rgb_bhw3.shape) == (1, 2, 3, 3)
    assert tuple(batch.prompt_bhw.shape) == (1, 192, 256)
    assert batch.confidence_bhw.shape == (1, 256, 192)
    assert_array_equal(batch.confidence_bhw[0, 0, :6], np.arange(6, dtype=np.uint8))
    assert batch.timestamps_ns == [100]
    second_batch = collate([
        {
            "video": torch.arange(18, dtype=torch.uint8).reshape(3, 3, 2),
            "depth": torch.zeros((1, 256, 192), dtype=torch.uint16),
            "conf": torch.arange(256 * 192).to(torch.uint8),
            "k": torch.eye(3, dtype=torch.float32).T.reshape(9),
            "pose_t": torch.tensor([1.0, 2.0, 3.0]),
            "pose_q": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        }
    ])
    assert second_batch is not None
    assert second_batch.timestamps_ns == [112]


def test_fuse_and_log_batch_preserves_frame_order_and_fusion_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log and fuse every completed row in timestamp order."""
    completed = CompletedPromptDABatch(
        timestamps_ns=[100, 200],
        quarter_turns=0,
        depth_mm_bhw=np.full((2, 4, 6), 1500, dtype=np.uint16),
        depth_model_mm_bhw=np.full((2, 2, 3), 1500, dtype=np.uint16),
        rgb_stored_bhw3=np.full((2, 4, 6, 3), 128, dtype=np.uint8),
        confidence_bhw=np.full((2, 1, 1), 2, dtype=np.uint8),
        K_native_b33=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        world_T_cam_b44=np.repeat(np.eye(4)[None], 2, axis=0),
        stored_hw=(4, 6),
    )
    logged_timestamps: list[int] = []
    fused_depths: list[np.ndarray] = []

    def record_depth(_recording: rr.RecordingStream, timestamp_ns: int, _depth_hw: np.ndarray) -> None:
        logged_timestamps.append(timestamp_ns)

    def record_fusion(_fuser: object, *, depth_hw: np.ndarray, **_kwargs: object) -> None:
        fused_depths.append(depth_hw.copy())

    monkeypatch.setattr("rerun_prompt_da.apis.prompt_da_arkitscenes.log_promptda_frame", record_depth)
    monkeypatch.setattr(Open3DFuser, "fuse_frames", record_fusion)
    fuser: Open3DFuser = object.__new__(Open3DFuser)
    recording = rr.RecordingStream("test", recording_id="segment", send_properties=False)

    inferred_frames = fuse_and_log_batch(completed, recording, fuser, PDAArkitScenesConfig().max_depth_range_meter)

    assert inferred_frames == 2
    assert logged_timestamps == [100, 200]
    assert len(fused_depths) == 2
    assert_array_equal(fused_depths[0], np.full((2, 3), 1500, dtype=np.uint16))


def test_stride_for_uses_nearest_native_frame_interval() -> None:
    """Choose the closest whole-frame stride without dropping below one."""
    assert stride_for(60.0, 10.0) == 6
    assert stride_for(60.0, 60.0) == 1
    assert stride_for(60.0, 7.0) == 9
    assert stride_for(60.0, 120.0) == 1


@pytest.mark.parametrize(
    ("quaternion", "expected_rotation"),
    [
        (np.array([0.0, 0.0, 0.0, 1.0]), np.eye(3)),
        (
            np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
    ],
)
def test_world_t_cam_from_pose_builds_rigid_transform(quaternion: np.ndarray, expected_rotation: np.ndarray) -> None:
    """Combine xyzw rotation and translation into a homogeneous pose."""
    transform = world_t_cam_from_pose(np.array([1.0, 2.0, 3.0]), quaternion)
    assert_allclose(transform[:3, :3], expected_rotation, atol=1e-7)
    assert_array_equal(transform[:3, 3], np.array([1.0, 2.0, 3.0]))
    assert_array_equal(transform[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_filter_depth_for_fusion_masks_low_confidence_and_far_depth() -> None:
    """Keep only medium-or-better depth within the fusion range."""
    depth = np.array([[1000, 2000, 3000, 5000], [1000, 2000, 3000, 4000], [1000, 2000, 3000, 4000], [1000, 2000, 3000, 4000]], dtype=np.uint16)
    confidence = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    filtered = filter_depth_for_fusion(depth, confidence, 4.0)
    assert_array_equal(
        filtered,
        np.array([[0, 0, 3000, 0], [0, 0, 3000, 4000], [1000, 2000, 3000, 4000], [1000, 2000, 3000, 4000]], dtype=np.uint16),
    )
    assert filtered.dtype == np.uint16


def test_segments_to_process_selects_explicit_segment_for_replacement() -> None:
    """Allow explicitly selected segments even when PromptDA already exists."""
    rows = [{"rerun_segment_id": "one", "rerun_layer_names": ["base", "promptda"]}, {"rerun_segment_id": "two", "rerun_layer_names": ["base"]}]
    assert segments_to_process(rows, "one", False, "promptda") == ["one"]


def test_segments_to_process_rejects_unknown_explicit_segment() -> None:
    """Show available ids when an explicit segment does not exist."""
    rows = [{"rerun_segment_id": "one", "rerun_layer_names": ["base"]}, {"rerun_segment_id": "two", "rerun_layer_names": ["base"]}]
    with pytest.raises(SystemExit, match="one.*two"):
        segments_to_process(rows, "missing", False, "promptda")


def test_segments_to_process_all_skips_existing_promptda_layers() -> None:
    """Process only segments that do not already carry the target layer."""
    rows = [{"rerun_segment_id": "one", "rerun_layer_names": ["base", "promptda"]}, {"rerun_segment_id": "two", "rerun_layer_names": ["base"]}]
    assert segments_to_process(rows, None, True, "promptda") == ["two"]


@pytest.mark.parametrize(("video_id", "process_all"), [(None, False), ("one", True)])
def test_segments_to_process_requires_exactly_one_selection_mode(video_id: str | None, process_all: bool) -> None:
    """Reject missing and ambiguous segment selection modes."""
    with pytest.raises(SystemExit, match="exactly one"):
        segments_to_process([], video_id, process_all, "promptda")
