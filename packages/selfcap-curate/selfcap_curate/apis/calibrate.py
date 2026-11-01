"""One-session calibration CLI; typed input adapters are kept at this boundary."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from dataforge.schema import video_path
from rerun.catalog import CatalogClient
from serde.json import to_json
from simplecv.catalog_video import read_catalog_videos

from selfcap_curate import catalog_io
from selfcap_curate.apis import open_session, session_output
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.detection_io import read_oak_observations, write_oak_egohand_layer
from selfcap_curate.device_tables import read_rotations, read_table
from selfcap_curate.device_vp import vp_focals
from selfcap_curate.exo_calibration import calibrate
from selfcap_curate.exo_landmarks import read_landmarks
from selfcap_curate.exo_refine import FocalSource
from selfcap_curate.exo_refine_catalog import evaluate, load_inputs, read_focals, usable_hand_frames
from selfcap_curate.factory import oak_centre, oak_stereo, read_quest_block
from selfcap_curate.hands_gpu import detect_episode, detect_oak_hands_gpu, pose_models, release_cuda_memory
from selfcap_curate.head_calibration import calibrate_with_head, head_track
from selfcap_curate.keypoints import read_ego_cameras
from selfcap_curate.keypoints_gpu import detect_keypoints, keypoint_models
from selfcap_curate.oak_egohand import detect_oak_egohand
from selfcap_curate.oak_mount import fixed_rotation, fixed_rotation_reason, gyro_mount, solve_oak
from selfcap_curate.quest_fit import read_session_episode
from selfcap_curate.records import OakFit, oak_quest_offset_ms


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
    dataset: str = catalog_io.DATASET_NAME
    """Dataset on ``catalog_url`` that holds the aligned bases (with their ``sapiens2`` layer)."""
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
    retain_models: bool = False
    """Keep inference engines between episodes in a long-lived batch worker."""
    exo_refine: bool = False
    """Opt into joint optical calibration when exo_landmarks is present."""
    exo_focal_source: FocalSource = "fit"
    """Use table focal, jointly fitted focal, or a fixed focal record."""
    exo_focal_file: Path | None = None
    """Typed JSON FocalRecord required by the fixed source."""
    exo_reference_dataset: str | None = None
    """Optional dataset on donor_catalog_url carrying independent RTMW gate observations."""
    exo_method: Literal["hands", "head"] = "hands"
    """Hands-only phone poses, or v3: extend and refine them with the wearer's head track (single-episode sessions)."""
    head_rate: float = 4.0
    """Dense detection rate (Hz) for the head-anchored fit."""


def main(config: Config, *, session_videos: catalog_io.SessionVideos | None = None) -> CalibSession:
    """Solve one session and write files only; never register or swap a layer."""
    session, keys, dataset = open_session(
        config.segment_ids, config.key_file, config.catalog_url, output=config.output_dir, protected=(), dataset=config.dataset
    )
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
        has_oak_clock = any(oak_quest_offset_ms(episode.sync) is not None for episode in episodes)
        rotation_reason: str | None = None if rotation is not None else fixed_rotation_reason(gyro, ego.device_id, rotations)
        if config.oak_free_rotation:
            rotation = None
            rotation_reason = "explicit free rotation"
        if use_egohand and has_oak_clock:
            print("OAK_ROTATION", "free" if rotation is None else "gyro", rotation_reason or "eligible gyro mount", flush=True)
            release_cuda_memory(models=not config.retain_models)
        for index, (episode, segment) in enumerate(zip(episodes, segments, strict=True)):
            if oak_quest_offset_ms(episode.sync) is None:
                print("OAK_UNPOSED", segment.id, "missing Quest/OAK clock", flush=True)
                if use_egohand:
                    write_oak_egohand_layer(output / "egohand" / f"{segment.id}.rrd", segment.id, episode.layout.oak_rig, episode.oak)
                continue
            if config.oak_observations is not None:
                oak = read_oak_observations(config.oak_observations / f"{segment.id}.parquet", segment.id)
            elif config.oak_method == "egohand":
                video = (segment.videos[video_path(episode.layout.oak_rig, 0)] if segment.videos is not None else
                         read_catalog_videos(dataset, segment.id, [video_path(episode.layout.oak_rig, 0)], "capture_time")[0])
                current = read_ego_cameras(dataset, dataset, segment.id, episode.layout.oak_rig, episode.layout.quest_rig)
                oak = detect_oak_egohand(episode, video, oak_centre(ego), rotation, config.oak_egohand_assets,
                                        camera=next((camera for camera in current if camera.cam == 0), None), retain_models=config.retain_models)
            else:
                oak = detect_oak_hands_gpu(dataset, episode.sync, episode.quest, episode.layout, pose_models()[1],
                                          video=None if segment.videos is None else segment.videos[video_path(episode.layout.oak_rig, 0)])
            episodes[index] = replace(episode, oak=oak)
            if use_egohand:
                write_oak_egohand_layer(output / "egohand" / f"{segment.id}.rrd", segment.id, episode.layout.oak_rig, oak)
        oak_fit = (
            solve_oak(session, episodes, oak_centre(ego), ego.device_id, rotation, weighted=use_egohand,
                      gate_image_speed=config.oak_observations is not None)
            if has_oak_clock else OakFit(session, False, ego.device_id, rotation_source="unposed: missing Quest/OAK clock",
                                         observation_source="egohand" if use_egohand else "sapiens")
        )
        if has_oak_clock:
            oak_fit = replace(oak_fit, rotation_source="free" if rotation is None else "gyro", rotation_reason=rotation_reason)
        # v3 uses the image-only vanishing-point focal of each device (median for unmeasured ones): the pooled
        # hand-profile table is ~25% short, because hands at one depth cannot separate focal from distance.
        focals = vp_focals([phone for episode in episodes for phone in episode.layout.phones]) if config.exo_method == "head" else read_table().focals
        state = CalibSession(
            session,
            calibrate(session, episodes, focals),
            oak_fit,
            read_quest_block(
                first_segment,
                factories["quest"],
                donors=None if config.donor_catalog_url is None else CatalogClient(config.donor_catalog_url).get_dataset(catalog_io.DATASET_NAME),
            ),
            oak_stereo(ego),
        )
        if config.exo_method == "head":
            if len(episodes) != 1:
                raise ValueError("head-anchored calibration handles one episode per session")
            release_cuda_memory(models=True)
            episode, segment = episodes[0], segments[0]
            quest_pose_ms = (episode.sync.offsets_ms or {}).get("quest_pose")
            if quest_pose_ms is None:
                print("HEAD_SKIP", segment.id, "no Quest clock", flush=True)
            else:
                # A phone align could not clock is detected on a guessed one; the head track then finds the real one.
                offsets_ms = dict(episode.sync.offsets_ms or {})
                known = [v for k, v in offsets_ms.items() if k.startswith("exo/") and v is not None]
                guess = float(np.median(known)) if known else quest_pose_ms
                provisional = frozenset(p for p in episode.layout.phones if offsets_ms.get("exo/" + p) is None)
                offsets_ms.update({"exo/" + p: guess for p in provisional})
                detection_sync = replace(episode.sync, offsets_ms=offsets_ms)
                dense = detect_keypoints(segment, *keypoint_models(), rate=config.head_rate, sync=detection_sync)
                track = head_track(segment.hands or catalog_io.read_catalog_hands(segment), quest_pose_ms)
                head = calibrate_with_head(state.exo, replace(episode, sync=detection_sync), dense, track, focals,
                                           {p: offsets_ms["exo/" + p] for p in episode.layout.phones}, provisional)
                hands_only = {p: c.camera.parameters.tolist() for p, c in state.exo.phones.items() if c.camera is not None}
                state = replace(state, exo=head.exo)
                (output / f"{session}.head.json").write_text(json.dumps({
                    "offsets_m": head.offsets.tolist(), "offsets_source": head.offsets_source, "head_convention": track.convention,
                    "measured_offsets_m": None if head.measured_offsets is None else head.measured_offsets.tolist(),
                    "hands_only_cameras": hands_only, "head_cameras": {p: c.camera.parameters.tolist() for p, c in head.exo.phones.items() if c.camera is not None},
                    "clock_corrections_ms": head.clock_corrections_ms, "provisional": sorted(provisional),
                    "phones": {p: asdict(r) for p, r in head.phones.items()}}, indent=1))
                if head.clock_corrections_ms:
                    print("HEAD_CLOCK", segment.id, head.clock_corrections_ms, flush=True)
                np.savez_compressed(output / f"{segment.id}.dense.npz", times=dense.times, **{
                    f"{field}__{p}": getattr(v, field) for p, v in dense.views.items() for field in ("boxes", "box_scores", "xy", "scores")})
                print("HEAD_CALIBRATION", segment.id, {p: (r.method, r.face_median_px) for p, r in head.phones.items()}, flush=True)
        if config.exo_refine:
            refinements = {}
            for segment in segments:
                labels = read_landmarks(dataset, segment.id)
                if labels is None:
                    continue
                posed = {rig: state.exo.phones[phone].camera for rig, phone in enumerate(segment.layout.phones)
                         if phone in state.exo.phones and state.exo.phones[phone].camera is not None}
                if len(posed) < 2:
                    continue
                rigs, seeds = tuple(posed), tuple(posed.values())
                inputs = load_inputs(segment, cameras=seeds, quest=state.quest, landmarks=labels, camera_rigs=rigs)
                # Hand frames are the refinement's cross-validation unit: a phone that never sees the hands (posed from
                # the head alone) keeps its pose and leaves the problem.
                hand_rigs = tuple(rig for view, rig in enumerate(rigs) if usable_hand_frames(inputs, view) >= 2)
                if len(hand_rigs) < 2:
                    continue
                if hand_rigs != rigs:
                    rigs, seeds = hand_rigs, tuple(posed[rig] for rig in hand_rigs)
                    inputs = load_inputs(segment, cameras=seeds, quest=state.quest, landmarks=labels, camera_rigs=rigs)
                if config.exo_reference_dataset is not None:
                    reference = CatalogClient(config.donor_catalog_url or config.catalog_url).get_dataset(name=config.exo_reference_dataset)
                    try:
                        reference_inputs = load_inputs(catalog_io.open_segment(reference, segment.id), cameras=seeds, landmarks=labels, camera_rigs=rigs)
                        inputs = replace(inputs, body=reference_inputs.body)
                    except ValueError as error:  # the reference lacks a clock for a phone this run posed: gate without body transfer
                        print("EXO_GATE_NO_BODY", segment.id, error, flush=True)
                fixed, digest = read_focals(config.exo_focal_file, inputs.phones)
                result, gate = evaluate(inputs, focal_source=config.exo_focal_source, fixed_focals=fixed, focal_record=digest)
                result = replace(result, camera_rigs=rigs)
                if gate.passes:  # a candidate that regresses any held-out metric stays a report; the input cameras stand
                    refinements[segment.id] = result
                (output / f"{segment.id}.exo-gate.json").write_text(to_json(gate))
                print("EXO_GATE", segment.id, "pass" if gate.passes else "candidate: fails no-regression gate", flush=True)
            state = replace(state, exo_refinements=refinements)
        for motion in episodes:
            write_calib_layer(output / motion.sync.episode_key.rrd_name, state, motion)
        (output / f"{session}.json").write_text(to_json(state))
        return state
