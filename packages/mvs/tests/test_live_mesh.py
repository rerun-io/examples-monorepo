import pytest

pytest.importorskip("rerun.catalog", reason="live mesh configuration imports need the Rerun catalog dependencies")
pytest.importorskip("rerun.experimental.dataloader", reason="live mesh configuration imports need the Rerun dataloader")
pytest.importorskip("torchcodec", reason="live mesh configuration imports need TorchCodec")

from mvs.catalog import CLOUD_CATALOG_URL, CatalogDataConfig  # noqa: E402


def test_catalog_data_config_defaults() -> None:
    """The default input configuration points at the cloud ARKitScenes dataset."""

    config = CatalogDataConfig()

    assert config.catalog_url == CLOUD_CATALOG_URL
    assert config.dataset_name == "arkitscenes"
    assert config.segments == ("42899799",)


def test_depth_raster_rgb_undoes_the_model_normalisation() -> None:
    """The fused colour is the model's input image, un-normalised and averaged 2x2 down to the depth raster."""
    import torch

    from mvs.apis.live_mesh import depth_raster_rgb
    from mvs.depth_engine import IMAGENET_MEAN, IMAGENET_STD

    rgb_3hw = torch.zeros(3, 384, 512)
    rgb_3hw[0] = 1.0  # red
    rgb_3hw[2, :, 256:] = 0.5  # blue on the right half
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rgb_hw3 = depth_raster_rgb(((rgb_3hw - mean) / std)[None])
    assert rgb_hw3.shape == (192, 256, 3) and rgb_hw3.dtype.name == "uint8"
    assert rgb_hw3[0, 0].tolist() == [255, 0, 0]
    assert rgb_hw3[0, 255].tolist() == [255, 0, 128]


def test_one_segment_per_run() -> None:
    """Two segments would restart the frame index on the same entity paths and mix their mesh histories."""
    from mvs.apis.live_mesh import single_segment

    assert single_segment(["41126369"]) == "41126369"
    for segment_ids in ([], ["41126369", "42899799"]):
        with pytest.raises(ValueError, match="one segment per recording"):
            single_segment(segment_ids)


def test_fusion_config_rejects_a_log_cadence_below_one() -> None:
    from mvs.apis.live_mesh import FusionConfig

    assert FusionConfig(log_every=1).log_every == 1
    with pytest.raises(ValueError, match="log_every"):
        FusionConfig(log_every=0)


def test_the_depth_stream_fetches_small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """One fetch block decodes all its full-resolution frames on the GPU at once, so the stream asks for small blocks."""
    from unittest.mock import MagicMock

    import torch
    from rerun.catalog import DatasetEntry
    from rerun.experimental.dataloader import RerunIterableDataset
    from simplecv.rerun_dataloader import SegmentNvdecDecoder

    import mvs.depth_stream
    from mvs.depth_stream import FETCH_BLOCK_SIZE, depth_dataset

    requested: dict[str, object] = {}

    # Subclasses, not mocks: beartype checks depth_dataset's annotated locals against these module names.
    class RecordingDataset(RerunIterableDataset):
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            requested.update(kwargs)

    class IdleDecoder(SegmentNvdecDecoder):
        def __init__(self, *args: object) -> None:
            del args

    monkeypatch.setattr(mvs.depth_stream, "RerunIterableDataset", RecordingDataset)
    monkeypatch.setattr(mvs.depth_stream, "SegmentNvdecDecoder", IdleDecoder)
    monkeypatch.setattr(mvs.depth_stream, "DataSource", lambda **kwargs: None)
    depth_dataset(MagicMock(spec=DatasetEntry), "42899799", torch.device("cpu"))

    assert FETCH_BLOCK_SIZE == 256
    assert requested["fetch_block_size"] == FETCH_BLOCK_SIZE
