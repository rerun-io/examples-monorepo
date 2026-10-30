"""One-session calibration CLI; typed input adapters are kept at this boundary."""

from dataclasses import dataclass, replace
from pathlib import Path

from dataforge.schema import video_path
from rerun.catalog import CatalogClient
from serde.json import to_json

from selfcap_curate import catalog_io
from selfcap_curate.apis import open_session, session_output
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.detection_io import read_oak_observations
from selfcap_curate.device_tables import read_rotations, read_table
from selfcap_curate.exo_calibration import calibrate
from selfcap_curate.factory import oak_centre, oak_stereo, read_quest_block
from selfcap_curate.hands_gpu import detect_episode, detect_oak_hands_gpu, pose_models
from selfcap_curate.oak_mount import fixed_rotation, gyro_mount, solve_oak
from selfcap_curate.quest_fit import read_session_episode


@dataclass
class Config:
    output_dir: Path
    """New session directory, published only when all outputs are complete."""
    segment_ids: tuple[str, ...] = ()
    """Explicit episodes of one session."""
    key_file: Path | None = None
    """Optional newline-separated segment ids or unprefixed keys."""
    catalog_url: str = catalog_io.CATALOG_URL
    """Read-only endpoint containing aligned bases and, unless donor_catalog_url is set, any healthy borrow source."""
    donor_catalog_url: str | None = None
    """Read-only endpoint holding the healthy camera-50 borrow sources when they are not on catalog_url."""
    oak_observations: Path | None = None
    """Optional directory of <segment>.parquet EgoHand pixels and box scores; enables weighted dense hand fitting."""
    oak_free_rotation: bool = False
    """Fit rotation freely for supplied EgoHand observations; otherwise preserve the production gyro rotation."""


def main(config: Config, *, session_videos: catalog_io.SessionVideos | None = None) -> CalibSession:
    """Solve one session and write files only; never register or swap a layer."""
    session, keys, dataset = open_session(config.segment_ids, config.key_file, config.catalog_url, output=config.output_dir, protected=())
    with session_output(config.output_dir) as output:
        episodes = []
        segments = []
        for key in keys:
            segment = catalog_io.open_segment(dataset, key.segment)
            if session_videos is not None:
                segment = replace(segment, videos=session_videos[key.segment].videos, hands=session_videos[key.segment].hands)
            segments.append(segment)
            sync = catalog_io.read_time_sync(dataset, key.segment)
            episode = read_session_episode(segment, sync)
            phones = catalog_io.read_phone_detections(segment)
            if set(episode.layout.phones) - phones.keys():
                phones = {**detect_episode(segment, episode.quest, *pose_models()), **phones}
            if config.oak_observations is not None:
                oak = read_oak_observations(config.oak_observations / f"{key.segment}.parquet", key.segment)
            else:
                oak = detect_oak_hands_gpu(
                    dataset,
                    sync,
                    episode.quest,
                    episode.layout,
                    pose_models()[1],
                    video=None if segment.videos is None else segment.videos[video_path(episode.layout.oak_rig, 0)],
                )
            episodes.append(replace(episode, detections=phones, oak=oak))
        first_segment = segments[0]
        factories = catalog_io.read_factories(first_segment)
        ego, gyro, rotations = factories["ego"], gyro_mount(session, episodes), read_rotations()
        rotation = fixed_rotation(gyro, ego.device_id, rotations)
        if config.oak_observations is not None:
            if config.oak_free_rotation:
                rotation = None
            elif rotation is None:
                raise ValueError("EgoHand gyro-fixed fitting requires an eligible gyro rotation; use oak_free_rotation explicitly to fit without it")
        state = CalibSession(
            session,
            calibrate(session, episodes, read_table().focals),
            solve_oak(session, episodes, oak_centre(ego), ego.device_id, rotation, weighted=config.oak_observations is not None),
            read_quest_block(
                first_segment,
                factories["quest"],
                donors=None if config.donor_catalog_url is None else CatalogClient(config.donor_catalog_url).get_dataset(catalog_io.DATASET_NAME),
            ),
            oak_stereo(ego),
        )
        for motion in episodes:
            write_calib_layer(output / motion.sync.episode_key.rrd_name, state, motion)
        (output / f"{session}.json").write_text(to_json(state))
        return state
