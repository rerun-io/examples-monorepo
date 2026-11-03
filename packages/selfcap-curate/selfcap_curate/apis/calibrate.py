"""One-session calibration CLI; typed input adapters are kept at this boundary."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from dataforge.datasets.selfcap_raw import oak_stereo
from dataforge.schema import video_path
from dataforge.writing import atomic_write
from rerun.catalog import CatalogClient
from serde import serde
from serde.json import to_json
from simplecv.catalog_video import read_catalog_videos

from selfcap_curate import catalog_io
from selfcap_curate.apis import open_session, session_output
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.detection_io import write_oak_egohand_layer
from selfcap_curate.device_tables import read_rotations, read_table
from selfcap_curate.device_vp import vp_focals
from selfcap_curate.exo_calibration import calibrate
from selfcap_curate.exo_landmarks import read_landmarks
from selfcap_curate.exo_refine import FocalSource
from selfcap_curate.exo_refine_catalog import evaluate, load_inputs, read_focals, usable_hand_frames
from selfcap_curate.factory import oak_centre, read_quest_block
from selfcap_curate.hands_gpu import detect_episode, pose_models, release_cuda_memory
from selfcap_curate.head_anchor import PhoneReport
from selfcap_curate.head_calibration import calibrate_with_head, head_track
from selfcap_curate.keypoints import read_ego_cameras
from selfcap_curate.oak_egohand import detect_oak_egohand, detect_oak_views
from selfcap_curate.oak_mount import fixed_rotation, fixed_rotation_reason, gyro_mount, oak_lenses, solve_oak, solve_oak_stereo
from selfcap_curate.quest_fit import read_session_episode
from selfcap_curate.records import OAK_CAMERAS, OakDetections, OakFit, oak_quest_offset_ms


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class HeadReport:
    """Head-anchored calibration provenance written beside the session record."""

    offsets_m: list[list[float]]
    """Face landmark offsets in the tracked head frame, in metres."""
    offsets_source: str
    """Source of the face landmark offsets."""
    head_convention: str
    """Coordinate convention of the source head track."""
    hands_only_cameras: dict[str, list[float]]
    """Phone camera parameters before head refinement."""
    head_cameras: dict[str, list[float]]
    """Phone camera parameters after head refinement."""
    clock_corrections_ms: dict[str, float]
    """Phone clock corrections in milliseconds."""
    provisional: list[str]
    """Phones whose initial clocks were guesses."""
    phones: dict[str, PhoneReport]
    """Per-phone fit and clock diagnostics."""


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
    oak_egohand_assets: Path | None = None
    """Optional local EgoHand root with weights/, engines/ and dinov3-bf16.onnx; None takes the pinned downloads and the trtkit cache."""
    exo_refine: bool = False
    """Opt into joint optical calibration when exo_landmarks is present."""
    exo_focal_source: FocalSource = "fit"
    """Use table focal, jointly fitted focal, or a fixed focal record."""
    exo_focal_file: Path | None = None
    """Typed JSON FocalRecord required by the fixed source."""
    exo_reference_dataset: str | None = None
    """Optional dataset on donor_catalog_url carrying independent RTMW gate observations."""
    exo_method: Literal["hands", "head"] = "head"
    """Hands-only phone poses, or v3: extend and refine them with the wearer's head track (single-episode sessions)."""


def main(config: Config) -> CalibSession:
    """Solve one session and write files only; never register or swap a layer."""
    session, keys, dataset = open_session(
        config.segment_ids, config.key_file, config.catalog_url, output=config.output_dir, protected=(), dataset=config.dataset
    )
    with session_output(config.output_dir) as output:
        episodes = []
        segments = []
        for key in keys:
            segment = catalog_io.open_segment(dataset, key.segment)
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
        has_oak_clock = any(oak_quest_offset_ms(episode.sync) is not None for episode in episodes)
        rotation_reason: str | None = None if rotation is not None else fixed_rotation_reason(gyro, ego.device_id, rotations)
        # Fit all OAK cameras jointly when stereo extrinsics are present; otherwise use RGB with the gyro mount.
        lenses = oak_lenses(ego)
        if has_oak_clock:
            print("OAK_ROTATION", "stereo+rgb joint" if lenses is not None else "free" if rotation is None else "gyro",
                  "" if lenses is not None else rotation_reason or "eligible gyro mount", flush=True)
            release_cuda_memory(models=True)
        views: dict[int, dict[str, OakDetections]] = {}
        for index, (episode, segment) in enumerate(zip(episodes, segments, strict=True)):
            if oak_quest_offset_ms(episode.sync) is None:
                print("OAK_UNPOSED", segment.id, "missing Quest/OAK clock", flush=True)
                write_oak_egohand_layer(output / "egohand" / f"{segment.id}.rrd", segment.id, episode.layout.oak_rig, {"rgb": episode.oak})
                continue
            if lenses is not None:
                oak_videos = read_catalog_videos(dataset, segment.id, [video_path(episode.layout.oak_rig, cam) for cam in range(len(OAK_CAMERAS))], "video_time")
                views[index] = detect_oak_views(episode, dict(zip(OAK_CAMERAS, oak_videos, strict=True)), lenses, config.oak_egohand_assets)
                oak = views[index]["rgb"]
            else:
                video = (segment.videos[video_path(episode.layout.oak_rig, 0)] if segment.videos is not None else
                         read_catalog_videos(dataset, segment.id, [video_path(episode.layout.oak_rig, 0)], "capture_time")[0])
                current = read_ego_cameras(dataset, dataset, segment.id, episode.layout.oak_rig, episode.layout.quest_rig)
                oak = detect_oak_egohand(episode, video, oak_centre(ego), rotation, config.oak_egohand_assets,
                                        camera=next((camera for camera in current if camera.cam == 0), None))
            episodes[index] = replace(episode, oak=oak)
            write_oak_egohand_layer(output / "egohand" / f"{segment.id}.rrd", segment.id, episode.layout.oak_rig, views.get(index, {"rgb": oak}))
        if not has_oak_clock:
            oak_fit = OakFit(session, False, ego.device_id, rotation_source="unposed: missing Quest/OAK clock",
                             observation_source="egohand")
        elif lenses is not None:
            oak_fit = solve_oak_stereo(session, [episodes[i] for i in views], list(views.values()), lenses, ego.device_id)
        else:
            oak_fit = replace(
                solve_oak(session, episodes, oak_centre(ego), ego.device_id, rotation, weighted=True, gate_image_speed=False),
                rotation_source="free" if rotation is None else "gyro", rotation_reason=rotation_reason,
            )
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
            episode, segment = episodes[0], segments[0]
            quest_pose_ms = (episode.sync.offsets_ms or {}).get("quest_pose")
            if quest_pose_ms is None:
                print("HEAD_SKIP", segment.id, "no Quest clock", flush=True)
            else:
                # The head fit reads align's detections (the episode's sapiens2 layer); a phone align could not clock
                # gets a guessed one, and the head track then finds the real one.
                offsets_ms = dict(episode.sync.offsets_ms or {})
                known = [v for k, v in offsets_ms.items() if k.startswith("exo/") and v is not None]
                guess = float(np.median(known)) if known else quest_pose_ms
                provisional = frozenset(p for p in episode.layout.phones if offsets_ms.get("exo/" + p) is None)
                offsets_ms.update({"exo/" + p: guess for p in provisional})
                detection_sync = replace(episode.sync, offsets_ms=offsets_ms)
                track = head_track(episode.hands, quest_pose_ms)
                head = calibrate_with_head(state.exo, replace(episode, sync=detection_sync), track, focals,
                                           {p: offsets_ms["exo/" + p] for p in episode.layout.phones}, provisional)
                hands_only = {p: c.camera.parameters.tolist() for p, c in state.exo.phones.items() if c.camera is not None}
                state = replace(state, exo=head.exo)
                with atomic_write(output / f"{session}.head.json") as staged:
                    staged.write_text(to_json(HeadReport(
                        head.offsets.tolist(), head.offsets_source, track.convention, hands_only,
                        {p: c.camera.parameters.tolist() for p, c in head.exo.phones.items() if c.camera is not None},
                        head.clock_corrections_ms, sorted(provisional), head.phones)))
                if head.clock_corrections_ms:
                    print("HEAD_CLOCK", segment.id, head.clock_corrections_ms, flush=True)
                print("HEAD_CALIBRATION", segment.id, {p: (r.method, r.face_median_px) for p, r in head.phones.items()}, flush=True)
        gates = {}  # segment id -> held-out gate, written into that segment's calib layer
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
                gates[segment.id] = gate
                print("EXO_GATE", segment.id, "pass" if gate.passes else "candidate: fails no-regression gate", flush=True)
            state = replace(state, exo_refinements=refinements)
        for motion in episodes:
            write_calib_layer(output / motion.sync.episode_key.rrd_name, state, motion, gate=gates.get(motion.sync.episode_key.segment))
        with atomic_write(output / f"{session}.json") as staged:
            staged.write_text(to_json(state))
        return state
