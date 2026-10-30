"""One-session calibration CLI; typed input adapters are kept at this boundary."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from dataforge.schema import video_path
from rerun.catalog import CatalogClient
from serde.json import to_json
from simplecv.catalog_video import read_catalog_videos

from selfcap_curate import catalog_io
from selfcap_curate.apis import open_session, session_output
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.detection_io import read_oak_observations, write_oak_egohand_layer
from selfcap_curate.device_tables import read_rotations, read_table
from selfcap_curate.exo_calibration import calibrate
from selfcap_curate.exo_landmarks import read_landmarks
from selfcap_curate.exo_refine import FocalSource
from selfcap_curate.exo_refine_catalog import evaluate, load_inputs, read_focals
from selfcap_curate.factory import oak_centre, oak_stereo, read_quest_block
from selfcap_curate.hands_gpu import detect_episode, detect_oak_hands_gpu, pose_models, release_cuda_memory
from selfcap_curate.keypoints import read_ego_cameras
from selfcap_curate.oak_egohand import detect_oak_egohand
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
    oak_method: Literal["egohand", "sapiens"] = "egohand"
    """OAK alignment observations; Sapiens retains the legacy 32-frame path for comparison."""
    oak_egohand_assets: Path = Path("data/egohand")
    """Local EgoHand asset root containing weights/, engines/ and dinov3-bf16.onnx."""
    exo_refine: bool = False
    """Opt into joint optical calibration when exo_landmarks is present."""
    exo_focal_source: FocalSource = "fit"
    """Use table focal, jointly fitted focal, or a fixed focal record."""
    exo_focal_file: Path | None = None
    """Typed JSON FocalRecord required by the fixed source."""
    exo_reference_dataset: str | None = None
    """Optional dataset on donor_catalog_url carrying independent RTMW gate observations."""


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
            episodes.append(replace(episode, detections=phones))
        first_segment = segments[0]
        factories = catalog_io.read_factories(first_segment)
        ego, gyro, rotations = factories["ego"], gyro_mount(session, episodes), read_rotations()
        rotation = fixed_rotation(gyro, ego.device_id, rotations)
        use_egohand = config.oak_observations is not None or config.oak_method == "egohand"
        if use_egohand:
            if config.oak_free_rotation:
                rotation = None
            elif rotation is None:
                raise ValueError("EgoHand gyro-fixed fitting requires an eligible gyro rotation; use oak_free_rotation explicitly to fit without it")
            release_cuda_memory(models=True)
        for index, (episode, segment) in enumerate(zip(episodes, segments, strict=True)):
            if config.oak_observations is not None:
                oak = read_oak_observations(config.oak_observations / f"{segment.id}.parquet", segment.id)
            elif config.oak_method == "egohand":
                video = (segment.videos[video_path(episode.layout.oak_rig, 0)] if segment.videos is not None else
                         read_catalog_videos(dataset, segment.id, [video_path(episode.layout.oak_rig, 0)], "capture_time")[0])
                current = read_ego_cameras(dataset, dataset, segment.id, episode.layout.oak_rig)
                sampling_rotation = fixed_rotation(gyro, ego.device_id, rotations)
                if sampling_rotation is None:
                    raise ValueError("EgoHand visibility sampling requires the gyro rotation")
                oak = detect_oak_egohand(episode, video, oak_centre(ego), sampling_rotation, config.oak_egohand_assets,
                                        camera=next((camera for camera in current if camera.cam == 0), None))
            else:
                oak = detect_oak_hands_gpu(dataset, episode.sync, episode.quest, episode.layout, pose_models()[1],
                                          video=None if segment.videos is None else segment.videos[video_path(episode.layout.oak_rig, 0)])
            episodes[index] = replace(episode, oak=oak)
            if use_egohand:
                write_oak_egohand_layer(output / "egohand" / f"{segment.id}.rrd", segment.id, episode.layout.oak_rig, oak)
        state = CalibSession(
            session,
            calibrate(session, episodes, read_table().focals),
            solve_oak(session, episodes, oak_centre(ego), ego.device_id, rotation, weighted=use_egohand,
                      gate_image_speed=config.oak_observations is not None),
            read_quest_block(
                first_segment,
                factories["quest"],
                donors=None if config.donor_catalog_url is None else CatalogClient(config.donor_catalog_url).get_dataset(catalog_io.DATASET_NAME),
            ),
            oak_stereo(ego),
        )
        if config.exo_refine:
            refinements = {}
            for segment in segments:
                labels = read_landmarks(dataset, segment.id)
                if labels is None:
                    continue
                seeds = []
                for phone in segment.layout.phones:
                    camera = state.exo.phones[phone].camera
                    if camera is None:
                        raise ValueError("exo refinement needs initialized cameras for every labelled view")
                    seeds.append(camera)
                inputs = load_inputs(segment, cameras=tuple(seeds), quest=state.quest, landmarks=labels)
                if config.exo_reference_dataset is not None:
                    reference = CatalogClient(config.donor_catalog_url or config.catalog_url).get_dataset(name=config.exo_reference_dataset)
                    reference_inputs = load_inputs(catalog_io.open_segment(reference, segment.id), landmarks=labels)
                    inputs = replace(inputs, body=reference_inputs.body)
                fixed, digest = read_focals(config.exo_focal_file, inputs.phones)
                result, gate = evaluate(inputs, focal_source=config.exo_focal_source, fixed_focals=fixed, focal_record=digest)
                refinements[segment.id] = result
                (output / f"{segment.id}.exo-gate.json").write_text(to_json(gate))
                print("EXO_GATE", segment.id, "pass" if gate.passes else "candidate: fails no-regression gate", flush=True)
            state = replace(state, exo_refinements=refinements)
        for motion in episodes:
            write_calib_layer(output / motion.sync.episode_key.rrd_name, state, motion)
        (output / f"{session}.json").write_text(to_json(state))
        return state
