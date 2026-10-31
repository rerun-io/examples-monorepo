"""Align, calibrate and keypoint one session in one process; local outputs only."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dataforge.timing import SequenceTimer
from rerun.catalog import CatalogClient
from rerun.server import Server

from selfcap_curate.apis import align, calibrate, keypoints
from selfcap_curate.catalog_io import CATALOG_URL, DATASET_NAME, SessionVideos, native_phone_rate
from selfcap_curate.exo_landmarks import read_landmarks, write_landmarks
from selfcap_curate.exo_refine import FocalSource
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
    dataset: str = DATASET_NAME
    """Dataset on ``catalog_url`` that holds the as-cut bases (with their ``gt`` layer)."""
    oak_observations: Path | None = None
    """Optional directory of dense EgoHand observation Parquets, passed to OAK calibration."""
    oak_free_rotation: bool = False
    """Fit the supplied EgoHand observations with free rotation instead of the production gyro rotation."""
    oak_method: Literal["egohand", "sapiens"] = "egohand"
    """Use 64 seeded EgoHand frames by default, or the legacy Sapiens OAK path."""
    oak_egohand_assets: Path = Path("data/egohand")
    """Local EgoHand weights and cached engine root, passed to calibration."""
    exo_refine: bool = False
    """Refine cameras from an exo_landmarks layer before triangulation."""
    exo_focal_source: FocalSource = "fit"
    """Table, joint-fit, or explicitly fixed phone focals."""
    exo_focal_file: Path | None = None
    """Fixed-focal JSON record, when requested."""
    exo_reference_dataset: str = DATASET_NAME
    """Read-only source of landmark labels and independent gate observations."""


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
                base_dir=config.base_dir, output_dir=aligned, segment_ids=config.segment_ids, key_file=config.key_file,
                catalog_url=config.catalog_url, dataset=config.dataset,
            ),
            session_videos=session_videos,
        )
    release_cuda_memory(models=config.oak_method == "egohand")
    with Server(host="127.0.0.1", port=0, datasets={DATASET_NAME: sorted(aligned.glob("*.rrd"))}) as server:
        detections: list[str] = [path.resolve().as_uri() for path in sorted((aligned / "sapiens2").glob("*.rrd"))]
        server.client().get_dataset(DATASET_NAME).register(detections, layer_name="sapiens2").wait()
        if config.exo_refine:
            reference = CatalogClient(config.catalog_url).get_dataset(name=config.exo_reference_dataset)
            for segment_id in session_videos:
                labels = read_landmarks(reference, segment_id)
                if labels is not None:
                    path = aligned / "exo_landmarks" / f"{segment_id}.rrd"
                    write_landmarks(path, segment_id, labels)
                    server.client().get_dataset(DATASET_NAME).register([path.resolve().as_uri()], layer_name="exo_landmarks").wait()
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
                    oak_method=config.oak_method,
                    oak_egohand_assets=config.oak_egohand_assets,
                    exo_refine=config.exo_refine,
                    exo_focal_source=config.exo_focal_source,
                    exo_focal_file=config.exo_focal_file,
                    exo_reference_dataset=config.exo_reference_dataset if config.exo_refine else None,
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
