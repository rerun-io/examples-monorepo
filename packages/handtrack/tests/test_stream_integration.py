"""One segment through the catalog stream (GPU + catalog), and the evaluation set's exact metadata."""

import pytest
import torch
from beartype.roar import BeartypeException

rr = pytest.importorskip("rerun", reason="needs rerun-sdk with the catalog extra")
pytest.importorskip("torchcodec", reason="needs torchcodec for NVDEC decode")

from handtrack.data.batches import CropKind, DetNetBatch, KeyNetBatch  # noqa: E402
from handtrack.data.catalog import CATALOG_URL, SHOW3D, UMETRACK  # noqa: E402
from handtrack.data.stream import CatalogStream, StreamConfig  # noqa: E402
from handtrack.eval.metrics import DetectionMetrics, KeypointMetrics, detection_metrics, keynet_metrics  # noqa: E402
from handtrack.geometry.letterbox import NET_HEIGHT, NET_WIDTH  # noqa: E402
from handtrack.labels.circles import square_boxes  # noqa: E402
from handtrack.train.source import DetNetValidation, KeyNetValidation  # noqa: E402

pytestmark = pytest.mark.integration

UMETRACK_SEGMENT: str = "umetrack__real__hand_hand__training__user_00__recording_01"
SHOW3D_SEGMENT: str = "show3d__PCW023__keyboard_inspecting_73f9"


def _require_resources(dataset: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("the stream decodes on NVDEC: no CUDA device")
    try:
        rr.catalog.CatalogClient(CATALOG_URL).get_dataset(dataset)
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {CATALOG_URL} dataset {dataset} unreachable: {error}")


def _check_detnet(batch: DetNetBatch) -> None:
    assert batch.pooled.shape[1:] == (1, 120, 160) and batch.pooled.dtype == torch.float32
    assert float(batch.pooled.min()) >= 0.0 and float(batch.pooled.max()) <= 1.0
    assert set(batch.presence.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(batch.circle_mask, batch.presence == 1.0)
    assert not (batch.circle_mask & ~batch.presence_mask).any()
    circles: torch.Tensor = batch.circle[batch.circle_mask]
    assert (circles[:, 2] > 0).all() and (circles[:, :2] > -0.5).all() and (circles[:, :2] < 1.5).all()
    assert batch.circle[~batch.circle_mask].eq(0).all()


def _check_keynet(batch: KeyNetBatch) -> None:
    assert batch.crops.shape[1:] == (1, 96, 96) and batch.keypoints.shape[1:] == (63,)
    assert batch.heatmaps.shape[1:] == (21, 18, 18) and batch.distance.shape[1:] == (21, 18)
    assert torch.equal(batch.positive, batch.kind == int(CropKind.POSITIVE))
    assert torch.equal(batch.presence == 1.0, batch.positive) and batch.presence_mask.all()
    # At least 17 of a positive's keypoints lie in the crop, so most heatmaps peak near 1; negatives carry none.
    peaks: torch.Tensor = batch.heatmaps[batch.positive].amax(dim=(-1, -2))
    assert ((peaks > 0.5).sum(dim=-1) >= 17).all()
    assert batch.heatmaps[~batch.positive].eq(0).all() and batch.distance[~batch.positive].eq(0).all()


@pytest.mark.parametrize(("dataset", "segment"), [(UMETRACK, UMETRACK_SEGMENT), (SHOW3D, SHOW3D_SEGMENT)])
def test_one_segment_through_the_stream(dataset: str, segment: str) -> None:
    _require_resources(dataset)
    config: StreamConfig = StreamConfig(
        datasets=(dataset,), segment_ids=(segment,), nets="both", producers=2, fetchers=1, detnet_buffer=2048, keynet_buffer=4096, detnet_batch_size=64, keynet_batch_size=64
    )
    with CatalogStream(config) as stream:
        stream.start_epoch(0)
        detnet_batches: int = 0
        while (batch := stream.next_detnet_batch()) is not None:
            _check_detnet(batch)
            detnet_batches += 1
        keynet_batches: int = 0
        while (crops := stream.next_keynet_batch()) is not None:
            _check_keynet(crops)
            keynet_batches += 1
        assert detnet_batches >= 2 and keynet_batches >= 2
        assert stream.stats.segments == 1 and stream.stats.images_decoded == stream.stats.detnet_samples
        # A second epoch decodes the segment again.
        stream.start_epoch(1)
        assert stream.next_detnet_batch() is not None


def test_evaluation_set_is_exact_and_repeatable() -> None:
    _require_resources(UMETRACK)
    config: StreamConfig = StreamConfig(
        datasets=(UMETRACK,), segment_ids=(UMETRACK_SEGMENT,), nets="both", producers=2, fetchers=1, detnet_batch_size=64, keynet_batch_size=64, validation=True
    )
    with CatalogStream(config) as stream:
        stream.start_epoch(0)
        first: DetNetBatch | None = stream.next_detnet_batch()
        assert first is not None
        metadata: DetNetValidation = stream.detnet_validation()
        # Ground-truth circles as predictions score perfect precision and recall under the paper's rule.
        circles: torch.Tensor = first.circle * first.circle.new_tensor([NET_WIDTH, NET_HEIGHT, NET_WIDTH]) + first.circle.new_tensor([0.0, 0.0, 1e-2])
        boxes: torch.Tensor = square_boxes(circles)
        count: int = first.pooled.shape[0]
        result: DetectionMetrics = detection_metrics(
            boxes.reshape(-1, 4),
            first.presence.reshape(-1),
            metadata.points.reshape(-1, 21, 2),
            metadata.in_front.reshape(-1, 21),
            metadata.camera.repeat_interleave(2),
            torch.arange(2, device=boxes.device).repeat(count),
            eligible=metadata.eligible.flatten(),
        )
        assert result.total.ground_truth > 0 and result.total.precision == 1.0 and result.total.recall == 1.0
        crops: KeyNetBatch | None = stream.next_keynet_batch()
        assert crops is not None
        exact: KeyNetValidation = stream.keynet_validation()
        scores: KeypointMetrics = keynet_metrics(
            exact.points_crop, exact.points_crop, exact.crop_from_net, exact.distance_mm, exact.distance_mm, crops.presence, crops.presence, crops.positive, crops.presence_mask
        )
        assert scores.keypoints > 0 and scores.pixel_error_sum == 0.0 and scores.presence.precision == 1.0
        # The same batches every epoch.
        stream.start_epoch(1)
        again: DetNetBatch | None = stream.next_detnet_batch()
        assert again is not None and torch.equal(again.pooled, first.pooled) and torch.equal(again.circle, first.circle)
