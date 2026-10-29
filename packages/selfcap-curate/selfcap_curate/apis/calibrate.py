"""One-session calibration CLI; typed input adapters are kept at this boundary."""

from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

from serde.json import to_json

from selfcap_curate import catalog_io
from selfcap_curate.apis import open_session, session_output
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
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
    """Read-only endpoint containing aligned bases and any healthy borrow source."""


def main(config: Config) -> CalibSession:
    """Solve one session and write files only; never register or swap a layer."""
    session, keys, dataset = open_session(config.segment_ids, config.key_file, config.catalog_url, output=config.output_dir, protected=())
    with session_output(config.output_dir) as output:
        models = cache(pose_models)
        episodes = []
        segments = []
        for key in keys:
            segment = catalog_io.open_segment(dataset, key.segment)
            segments.append(segment)
            sync = catalog_io.read_time_sync(dataset, key.segment)
            episode = read_session_episode(segment, sync)
            phones = catalog_io.read_phone_detections(segment)
            if set(episode.layout.phones) - phones.keys():
                phones = {**detect_episode(segment, episode.quest, *models()), **phones}
            oak = detect_oak_hands_gpu(dataset, sync, episode.quest, episode.layout, models()[1])
            episodes.append(replace(episode, detections=phones, oak=oak))
        first_segment = segments[0]
        factories = catalog_io.read_factories(first_segment)
        ego, gyro, rotations = factories["ego"], gyro_mount(session, episodes), read_rotations()
        state = CalibSession(
            session,
            calibrate(session, episodes, read_table().focals),
            solve_oak(session, episodes, oak_centre(ego), ego.device_id, fixed_rotation(gyro, ego.device_id, rotations)),
            read_quest_block(first_segment, factories["quest"]),
            oak_stereo(ego),
        )
        for motion in episodes:
            write_calib_layer(output / motion.sync.episode_key.rrd_name, state, motion)
        (output / f"{session}.json").write_text(to_json(state))
        return state
