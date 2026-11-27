"""The 0.42 dataloader's luma plane gives the same grey as the old NVDEC RGB path on the grey cameras of the base mix."""

import pytest
import torch
from beartype.roar import BeartypeException
from jaxtyping import UInt8
from torch import Tensor

pytestmark = pytest.mark.integration

rr = pytest.importorskip("rerun", reason="needs rerun-sdk")
dataloader = pytest.importorskip("rerun.experimental.dataloader", reason="needs the rerun dataloader extra")

from simplecv.catalog_video_reader import open_decoders  # noqa: E402

from handtrack.data.catalog import CATALOG_URL  # noqa: E402
from handtrack.data.luma import grey_from_luma  # noqa: E402

FRAMES: int = 5


def _dataset(name: str):
    try:
        return rr.catalog.CatalogClient(CATALOG_URL).get_dataset(name)
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {CATALOG_URL} dataset {name} unreachable: {error}")


@pytest.mark.parametrize(("name", "entity"), [("dataforge-umetrack", "/world/rig_00/cam_00/pinhole/video"), ("dataforge-show3d", "/world/rig_01/cam_00/pinhole/video")])
def test_luma_grey_matches_the_rgb_decode_frame_by_frame(name: str, entity: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("the old path decodes on NVDEC")
    dataset = _dataset(name)
    segment: str = sorted(dataset.segment_ids())[0]
    video = open_decoders(dataset, segment, [entity], "video_time", torch.device("cuda"))[0]
    old: UInt8[Tensor, "n h w"] = video.get_frames_at(list(range(FRAMES)))[:, 0].cpu()
    stream = dataloader.RerunIterableDataset(
        dataloader.DataSource(dataset=dataset, segments=[segment]),
        index="frame_index",
        fields={"video": dataloader.Field(f"{entity}:VideoStream:sample", decode=dataloader.VideoFrameDecoder(codec=video.index.codec, output_format="yuv420p"))},
        shuffle_strategy=dataloader.NoShuffle(),
    )
    frames = [sample["video"] for _, sample in zip(range(FRAMES), stream, strict=False)]
    full = torch.tensor([str(frame.color_range).lower().endswith("full") for frame in frames])
    new: UInt8[Tensor, "n h w"] = grey_from_luma(torch.stack([frame.y[0] for frame in frames]), full)
    assert int((old.int() - new.int()).abs().max()) <= 1
