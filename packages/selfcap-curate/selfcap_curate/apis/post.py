"""Align, calibrate and keypoint one session in one process; local outputs only."""

from dataclasses import dataclass
from pathlib import Path

from dataforge.timing import SequenceTimer
from rerun.server import Server

from selfcap_curate.apis import align, calibrate, keypoints
from selfcap_curate.catalog_io import CATALOG_URL, DATASET_NAME


@dataclass
class Config:
    """One session's as-cut bases in; every layer an operator registers out."""

    base_dir: Path
    """As-cut base RRD directory with presented timestamps, read only."""
    output_dir: Path
    """New directory: align/ holds the aligned bases, sapiens2/ and sync records; keypoints/ holds the calib layers."""
    rate: float
    """Keypoints sampling frequency in Hz on the aligned clock."""
    segment_ids: tuple[str, ...] = ()
    """Explicit selfcap__date__sid8__epNNN ids of one session."""
    key_file: Path | None = None
    """Optional text file, one segment id or unprefixed key per line."""
    catalog_url: str = CATALOG_URL
    """Catalog with the as-cut bases and any camera-50 donor; never registered or swapped here."""


def main(config: Config) -> None:
    """Run the three stages in one process, so imports and models load once.

    Calibrate and keypoints read align's output from a private in-process server, exactly the files an operator
    registers afterwards, so nothing is registered between the stages.
    """
    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise FileExistsError(config.output_dir)
    aligned: Path = config.output_dir / "align"
    timer = SequenceTimer()
    with timer.stage("align"):
        align.main(
            align.Config(
                base_dir=config.base_dir, output_dir=aligned, segment_ids=config.segment_ids, key_file=config.key_file, catalog_url=config.catalog_url
            )
        )
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
                )
            )
        with timer.stage("keypoints"):
            keypoints.main(
                keypoints.Config(
                    calib_dir=config.output_dir / "calib",
                    output_dir=config.output_dir / "keypoints",
                    rate=config.rate,
                    segment_ids=config.segment_ids,
                    key_file=config.key_file,
                    catalog_url=server.url(),
                )
            )
    print("POST_TIME", timer.stage_s, flush=True)
