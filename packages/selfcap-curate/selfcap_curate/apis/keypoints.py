"""One-session catalog GPU keypoints; local calibration owns the output."""

from dataclasses import dataclass
from pathlib import Path

from dataforge.timing import SequenceTimer
from rerun.chunk import Chunk
from rerun.server import Server
from simplecv.catalog_video import CatalogVideo

from selfcap_curate.apis import open_session
from selfcap_curate.camera_centers import CenterTrack, camera_center_chunks, read_quest_centers
from selfcap_curate.catalog_io import (
    CATALOG_URL,
    DATASET_NAME,
    Segment,
    aligned_phone_videos,
    median_rate,
    open_segment,
    read_calibration_sync,
    read_catalog_hands,
    read_phone_videos,
    read_time_sync,
)
from selfcap_curate.keypoints import (
    Camera,
    EgoCamera,
    PoseModel,
    ego_quest_hands,
    read_anchors,
    read_cameras,
    read_ego_cameras,
    smooth_keypoints,
    triangulate_episode,
    write_keypoints,
)
from selfcap_curate.keypoints_gpu import detect_keypoints, keypoint_models
from selfcap_curate.records import CatalogHands, EpisodeKey, TimeSync


@dataclass
class Config:
    calib_dir: Path  # Read-only selfcap-calibrate files whose camera values own the output.
    output_dir: Path  # Per-episode keypoints layers (calibration + keypoints). Existing targets refused.
    rate: float | None = None
    """Common aligned sampling frequency in Hz; None uses each episode's median native phone rate."""
    segment_ids: tuple[str, ...] = ()  # Explicit segment ids or unprefixed episode keys of one session.
    key_file: Path | None = None  # Optional newline-separated ids, combined with segment_ids.
    catalog_url: str = CATALOG_URL  # Read-only catalog containing patched bases, videos, clocks and anchors.
    dataset: str = DATASET_NAME  # Dataset on catalog_url holding those bases, as in calibrate.
    model: PoseModel = "rtmw"
    """2D estimator: RTMW-x by default; sapiens2 selects Sapiens2 0.4B NVFP4 and writes its keypoints under sapiens2/ instead."""


@dataclass(frozen=True, slots=True)
class KeypointEpisode:
    key: EpisodeKey
    """Canonical episode and output names."""
    segment: Segment
    """Base segment and its discovered physical layout."""
    videos: dict[str, CatalogVideo]
    """Phone packets already read for centres and native-rate measurement."""
    hands: CatalogHands | None
    """Normalized capture-clock Quest samples, absent when the Quest clock is missing."""
    sync: TimeSync
    """Validated alignment record."""
    rate: float
    """Aligned sampling frequency in Hz."""
    cameras: dict[str, Camera]
    """Calibrated phones with usable offsets."""
    ego: tuple[EgoCamera, ...]
    """OAK cameras with a calibrated mount; overlays are projected through their full lens model."""
    center_chunks: tuple[Chunk, ...]
    """Camera-centre overlays sampled at each exo video frame time."""


def main(config: Config) -> None:
    """Infer then triangulate into one keypoints layer per episode. Never register or swap."""
    _, keys, dataset = open_session(
        config.segment_ids,
        config.key_file,
        config.catalog_url,
        output=config.output_dir,
        protected=(config.calib_dir,),
        source=config.calib_dir,
        new_directory=False,
        rate=config.rate,
        dataset=config.dataset,
    )
    processable: list[KeypointEpisode] = []
    files = [config.calib_dir / key.rrd_name for key in keys]
    with Server(host="127.0.0.1", port=0, datasets={DATASET_NAME: files}) as server:
        calibration = server.client().get_dataset(DATASET_NAME)
        for key in keys:
            segment = open_segment(dataset, key.segment)
            try:
                cameras = read_cameras(calibration, segment.id, segment.layout.phones)
            except ValueError as error:
                print("KEYPOINTS_SKIP", key, str(error), flush=True)
                continue
            sync = read_calibration_sync(calibration, segment.id, read_time_sync(dataset, segment.id))
            cameras = {p: c for p, c in cameras.items() if (sync.offsets_ms or {}).get("exo/" + p) is not None}
            if len(cameras) < 2:
                print("KEYPOINTS_SKIP", key, "fewer than two calibrated and synced phones", flush=True)
            else:
                ego: tuple[EgoCamera, ...] = tuple(read_ego_cameras(calibration, dataset, segment.id, segment.layout.oak_rig, segment.layout.quest_rig))
                quest: list[CenterTrack] = read_quest_centers(calibration, dataset, segment.id, segment.layout.quest_rig)
                videos: dict[str, CatalogVideo] = aligned_phone_videos(read_phone_videos(segment), sync)
                centers: tuple[Chunk, ...] = tuple(camera_center_chunks(cameras, ego, quest, {phone: video.t_ns for phone, video in videos.items()}))
                # A phone's video_time is its capture_time plus one constant, so these packets hold the native spacing: no second read.
                rate: float = config.rate if config.rate is not None else median_rate(videos)
                print("KEYPOINTS_RATE", key, rate, flush=True)
                processable.append(KeypointEpisode(key, segment, videos, read_catalog_hands(segment) if (sync.offsets_ms or {}).get("quest_pose") is not None else None, sync, rate, cameras, ego, centers))
    if not processable:
        return
    detector, pose = keypoint_models(config.model)
    timer = SequenceTimer()
    for episode in processable:
        key = episode.key
        with timer.stage(f"{key}/detect"):
            detections = detect_keypoints(episode.segment, detector, pose, rate=episode.rate, sync=episode.sync, videos=episode.videos)
        with timer.stage(f"{key}/triangulate"):
            result = smooth_keypoints(triangulate_episode(detections, episode.cameras, read_anchors(episode.hands, detections.times, episode.sync)))
        with timer.stage(f"{key}/write"):
            write_keypoints(
                config.calib_dir / key.rrd_name,
                config.output_dir / key.rrd_name,
                result,
                ego=episode.ego,
                quest_hands=ego_quest_hands(episode.hands, episode.sync),
                camera_center_chunks=episode.center_chunks,
                model=config.model,
            )
        print(
            "KEYPOINTS_TIME",
            key,
            len(result.times),
            int((result.n_views >= 2).sum()),
            {k: v for k, v in timer.stage_s.items() if k.startswith(f"{key}/")},
            flush=True,
        )
