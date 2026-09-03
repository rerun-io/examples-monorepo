import rerun as rr
from rerun.experimental.dataloader import DataSource, Field, FixedRateSampling, NoShuffle, RerunIterableDataset, VideoFrameDecoder
from torch import Tensor

if __name__ == "__main__":
    rr.init("dataloader", spawn=True)
    client = rr.catalog.CatalogClient(url="rerun+http://127.0.0.1:9991")

    source = DataSource(
        dataset=client.get_dataset("hocap"),
        segments=[
            "hocap__subject_7__20231023_163653",
        ],
    )
    print(source)

    fields = {
        "video": Field(
            "/world/exo/037522251142/pinhole/video:VideoStream:sample",
            decode=VideoFrameDecoder(codec="h264"),
        ),
    }

    ds = RerunIterableDataset(
        source=source,
        index="video_time",
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=15.0),
        shuffle_strategy=NoShuffle(),
    )
    for i, item in enumerate(ds):
        rr.set_time(timeline="test", sequence=i)
        video = item["video"]
        if video is None:
            print(f"Item {i}: video is None")
            continue
        if not isinstance(video, Tensor):
            raise TypeError(f"Expected an RGB tensor, got {type(video).__name__}")
        print(f"Item {i}: {item}")
        print(video.shape)
        video_frame = video.permute(1, 2, 0).numpy()
        rr.log("video_frame", rr.Image(video_frame))
