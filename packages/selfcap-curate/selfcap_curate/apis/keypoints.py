"""One-session catalog GPU keypoints; local calibration owns the output."""

from dataclasses import dataclass, replace
from pathlib import Path

from dataforge.timing import SequenceTimer
from rerun.chunk import Chunk
from rerun.server import Server
from simplecv.catalog_video import CatalogVideo

from selfcap_curate.apis import open_session
from selfcap_curate.camera_centers import CenterTrack, camera_center_chunks, read_quest_centers
from selfcap_curate.catalog_io import CATALOG_URL, DATASET_NAME, Segment, SessionVideos, open_segment, read_phone_videos, read_time_sync
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
from selfcap_curate.records import EpisodeKey, TimeSync


@dataclass
class Config:
    calib_dir: Path  # Read-only selfcap-calibrate files whose camera values own the output.
    output_dir: Path  # Per-episode outputs; Sapiens2 contains only its new keypoint layer. Existing targets refused.
    rate: float  # Explicit common aligned sampling frequency in Hz.
    segment_ids: tuple[str, ...] = ()  # Explicit segment ids or unprefixed episode keys of one session.
    key_file: Path | None = None  # Optional newline-separated ids, combined with segment_ids.
    catalog_url: str = CATALOG_URL  # Read-only catalog containing patched bases, videos, clocks and anchors.
    model: PoseModel = "rtmw"
    """2D estimator; Sapiens2 writes an isolated layer without calibration chunks."""


@dataclass(frozen=True, slots=True)
class KeypointEpisode:
    key: EpisodeKey
    """Canonical episode and output names."""
    segment: Segment
    """Base segment and its discovered physical layout."""
    sync: TimeSync
    """Validated alignment record."""
    cameras: dict[str, Camera]
    """Calibrated phones with usable offsets."""
    ego: tuple[EgoCamera, ...]
    """OAK cameras with a calibrated mount; overlays are projected through their full lens model."""
    center_chunks: tuple[Chunk, ...]
    """Camera-centre overlays sampled at each exo video frame time."""


def main(config: Config, *, session_videos: SessionVideos | None = None) -> None:
    """Infer then triangulate; Sapiens2 emits a separate layer. Never register or swap."""
    _, keys, dataset = open_session(
        config.segment_ids,
        config.key_file,
        config.catalog_url,
        output=config.output_dir,
        protected=(config.calib_dir,),
        source=config.calib_dir,
        new_directory=False,
        rate=config.rate,
    )
    processable: list[KeypointEpisode] = []
    files = [config.calib_dir / key.rrd_name for key in keys]
    with Server(host="127.0.0.1", port=0, datasets={DATASET_NAME: files}) as server:
        calibration = server.client().get_dataset(DATASET_NAME)
        for key in keys:
            segment = open_segment(dataset, key.segment)
            if session_videos is not None:
                segment = replace(segment, videos=session_videos[key.segment].videos, hands=session_videos[key.segment].hands)
            try:
                cameras = read_cameras(calibration, segment.id, segment.layout.phones)
            except ValueError as error:
                print("KEYPOINTS_SKIP", key, str(error), flush=True)
                continue
            sync = read_time_sync(dataset, segment.id)
            cameras = {p: c for p, c in cameras.items() if (sync.offsets_ms or {}).get("exo/" + p) is not None}
            if len(cameras) < 2:
                print("KEYPOINTS_SKIP", key, "fewer than two calibrated and synced phones", flush=True)
            else:
                ego: tuple[EgoCamera, ...] = tuple(read_ego_cameras(calibration, dataset, segment.id, segment.layout.oak_rig))
                quest: list[CenterTrack] = read_quest_centers(calibration, dataset, segment.id, segment.layout.quest_rig)
                videos: dict[str, CatalogVideo] = read_phone_videos(segment, timeline="video_time", sync=sync)
                centers: tuple[Chunk, ...] = tuple(camera_center_chunks(cameras, ego, quest, {phone: video.t_ns for phone, video in videos.items()}))
                processable.append(KeypointEpisode(key, segment, sync, cameras, ego, centers))
    if not processable:
        return
    detector, pose = keypoint_models() if config.model == "rtmw" else keypoint_models(config.model)
    timer = SequenceTimer()
    for episode in processable:
        key = episode.key
        with timer.stage(f"{key}/detect"):
            detections = detect_keypoints(episode.segment, detector, pose, rate=config.rate, sync=episode.sync)
        with timer.stage(f"{key}/triangulate"):
            result = smooth_keypoints(triangulate_episode(detections, episode.cameras, read_anchors(episode.segment, detections.times, episode.sync)))
        with timer.stage(f"{key}/write"):
            write_keypoints(
                config.calib_dir / key.rrd_name,
                config.output_dir / key.rrd_name,
                result,
                ego=episode.ego,
                quest_hands=ego_quest_hands(episode.segment, episode.sync),
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
