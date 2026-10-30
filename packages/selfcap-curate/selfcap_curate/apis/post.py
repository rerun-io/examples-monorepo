"""Align, calibrate and keypoint one session in one process; local outputs only."""

from dataclasses import dataclass
from pathlib import Path

from dataforge.timing import SequenceTimer
from rerun.server import Server

from selfcap_curate.apis import align, calibrate, keypoints
from selfcap_curate.catalog_io import CATALOG_URL, DATASET_NAME, SessionVideos, native_phone_rate
from selfcap_curate.hands_gpu import release_cuda_memory


@dataclass
class Config:
    """One session's as-cut bases in; every layer an operator registers out."""

    base_dir: Path
    """As-cut base RRD directory with presented timestamps, read only."""
    output_dir: Path
    """New directory: align/ holds the aligned bases, sapiens2/ and sync records; keypoints/ holds the calib layers."""
    rate: float | None = None
    """Keypoints frequency in Hz; omitted uses the first episode's median native phone rate."""
    segment_ids: tuple[str, ...] = ()
    """Explicit selfcap__date__sid8__epNNN ids of one session."""
    key_file: Path | None = None
    """Optional text file, one segment id or unprefixed key per line."""
    catalog_url: str = CATALOG_URL
    """Catalog with the as-cut bases and any camera-50 donor; never registered or swapped here."""
    oak_observations: Path | None = None
    """Optional directory of dense EgoHand observation Parquets, passed to OAK calibration."""
    oak_free_rotation: bool = False
    """Fit the supplied EgoHand observations with free rotation instead of the production gyro rotation."""


def main(config: Config) -> None:
    """Run the three stages in one process, so imports and models load once.

    Calibrate and keypoints read align's output from a private in-process server, exactly the files an operator
    registers afterwards, so nothing is registered between the stages.
    """
    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise FileExistsError(config.output_dir)
    aligned: Path = config.output_dir / "align"
    session_videos: SessionVideos = {}
    timer = SequenceTimer()
    with timer.stage("align"):
        align.main(
            align.Config(
                base_dir=config.base_dir, output_dir=aligned, segment_ids=config.segment_ids, key_file=config.key_file, catalog_url=config.catalog_url
            ),
            session_videos=session_videos,
        )
    release_cuda_memory(models=False)  # align's decode and links buffers; calibrate still needs Sapiens2
    with Server(host="127.0.0.1", port=0, datasets={DATASET_NAME: sorted(aligned.glob("*.rrd"))}) as server:
        detections: list[str] = [path.resolve().as_uri() for path in sorted((aligned / "sapiens2").glob("*.rrd"))]
        server.client().get_dataset(DATASET_NAME).register(detections, layer_name="sapiens2").wait()
        with timer.stage("calibrate"):
            calibrate.main(
                calibrate.Config(
                    output_dir=config.output_dir / "calib",
                    segment_ids=config.segment_ids,
                    key_file=config.key_file,
                    catalog_url=server.url(),
                    donor_catalog_url=config.catalog_url,
                    oak_observations=config.oak_observations,
                    oak_free_rotation=config.oak_free_rotation,
                ),
                session_videos=session_videos,
            )
        release_cuda_memory(models=True)  # keypoints never uses Sapiens2: free its engines so sessions can share the GPU
        with timer.stage("keypoints"):
            keypoints.main(
                keypoints.Config(
                    calib_dir=config.output_dir / "calib",
                    output_dir=config.output_dir / "keypoints",
                    rate=config.rate if config.rate is not None else native_phone_rate(next(iter(session_videos.values()))),
                    segment_ids=config.segment_ids,
                    key_file=config.key_file,
                    catalog_url=server.url(),
                ),
                session_videos=session_videos,
            )
    print("POST_TIME", timer.stage_s, flush=True)
