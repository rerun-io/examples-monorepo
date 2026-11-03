"""One-session calibration CLI; typed input adapters are kept at this boundary."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from dataforge.datasets.selfcap_raw import oak_stereo
from dataforge.oak_mount import FactoryCalibration
from dataforge.schema import video_path
from dataforge.writing import atomic_write
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from serde import serde
from serde.json import to_json
from simplecv.catalog_video import read_catalog_videos

from selfcap_curate import catalog_io
from selfcap_curate.apis import open_session, session_output
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.detection_io import write_oak_egohand_layer
from selfcap_curate.device_tables import read_rotations, read_table, vp_focals
from selfcap_curate.exo_calibration import calibrate
from selfcap_curate.factory import oak_centre, read_quest_block
from selfcap_curate.hands_gpu import detect_episode, pose_models, release_cuda_memory
from selfcap_curate.head_anchor import PhoneReport
from selfcap_curate.head_calibration import calibrate_with_head, head_track
from selfcap_curate.keypoints import read_ego_cameras
from selfcap_curate.oak_egohand import detect_oak_egohand, detect_oak_views
from selfcap_curate.oak_mount import OakLens, fixed_rotation, fixed_rotation_reason, gyro_mount, oak_lenses, solve_oak, solve_oak_stereo
from selfcap_curate.quest_fit import read_session_episode
from selfcap_curate.records import OAK_CAMERAS, EpisodeKey, OakDetections, OakFit, SessionEpisode, oak_quest_offset_ms


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
    exo_method: Literal["hands", "head"] = "head"
    """Hands-only phone poses, or v3: extend and refine them with the wearer's head track (single-episode sessions)."""


OakMode: TypeAlias = Literal["unposed", "stereo", "rgb"]


def read_episodes(dataset: DatasetEntry, keys: tuple[EpisodeKey, ...]) -> tuple[list[SessionEpisode], list[catalog_io.Segment]]:
    """Load each episode's CPU inputs once and reuse its catalog phone detections."""
    episodes: list[SessionEpisode] = []
    segments: list[catalog_io.Segment] = []
    for key in keys:
        segment = catalog_io.open_segment(dataset, key.segment)
        segments.append(segment)
        episode = read_session_episode(segment, catalog_io.read_time_sync(dataset, key.segment))
        phones = catalog_io.read_phone_detections(segment)
        if set(episode.layout.phones) - phones.keys():
            phones = {**detect_episode(segment, episode.quest, *pose_models()), **phones}
        episodes.append(replace(episode, detections=phones))
    return episodes, segments


def detect_oak_episode(
    episode: SessionEpisode, segment: catalog_io.Segment, ego: FactoryCalibration,
    lenses: dict[str, OakLens] | None, rotation: Float64[ndarray, "3 3"] | None, assets: Path | None,
) -> dict[str, OakDetections]:
    """Infer the available OAK views on the clock used by each calibration path."""
    if lenses is not None:
        videos = read_catalog_videos(
            segment.dataset, segment.id, [video_path(episode.layout.oak_rig, cam) for cam in range(len(OAK_CAMERAS))], "video_time"
        )
        return detect_oak_views(episode, dict(zip(OAK_CAMERAS, videos, strict=True)), lenses, assets)
    video = read_catalog_videos(segment.dataset, segment.id, [video_path(episode.layout.oak_rig, 0)], "capture_time")[0]
    current = read_ego_cameras(segment.dataset, segment.dataset, segment.id, episode.layout.oak_rig, episode.layout.quest_rig)
    return {"rgb": detect_oak_egohand(
        episode, video, oak_centre(ego), rotation, assets, camera=next((camera for camera in current if camera.cam == 0), None)
    )}


def fit_oak_mount(
    session: str, episodes: list[SessionEpisode], segments: list[catalog_io.Segment], ego: FactoryCalibration,
    output: Path, assets: Path | None,
) -> tuple[OakFit, list[SessionEpisode]]:
    """Choose one mount mode, write its observations, and fit the session mount."""
    gyro, rotations = gyro_mount(session, episodes), read_rotations()
    rotation = fixed_rotation(gyro, ego.device_id, rotations)
    reason = None if rotation is not None else fixed_rotation_reason(gyro, ego.device_id, rotations)
    lenses = oak_lenses(ego)
    mode: OakMode = "unposed" if not any(oak_quest_offset_ms(e.sync) is not None for e in episodes) else "stereo" if lenses is not None else "rgb"
    if mode != "unposed":
        print("OAK_ROTATION", "stereo+rgb joint" if mode == "stereo" else "free" if rotation is None else "gyro",
              "" if mode == "stereo" else reason or "eligible gyro mount", flush=True)
        release_cuda_memory(models=True)
    updated: list[SessionEpisode] = []
    posed: list[SessionEpisode] = []
    views: list[dict[str, OakDetections]] = []
    for episode, segment in zip(episodes, segments, strict=True):
        if oak_quest_offset_ms(episode.sync) is None:
            print("OAK_UNPOSED", segment.id, "missing Quest/OAK clock", flush=True)
            detections = {"rgb": episode.oak}
        else:
            detections = detect_oak_episode(episode, segment, ego, lenses, rotation, assets)
            episode = replace(episode, oak=detections["rgb"])
            posed.append(episode)
            views.append(detections)
        updated.append(episode)
        write_oak_egohand_layer(output / "egohand" / f"{segment.id}.rrd", segment.id, episode.layout.oak_rig, detections)
    if mode == "unposed":
        fit = OakFit(session, False, ego.device_id, rotation_source="unposed: missing Quest/OAK clock", observation_source="egohand")
    elif mode == "stereo":
        assert lenses is not None
        fit = solve_oak_stereo(session, posed, views, lenses, ego.device_id)
    else:
        fit = replace(solve_oak(session, updated, oak_centre(ego), ego.device_id, rotation),
                      rotation_source="free" if rotation is None else "gyro", rotation_reason=reason)
    return fit, updated


def refine_with_head(state: CalibSession, episode: SessionEpisode, focals: dict[str, float], output: Path) -> CalibSession:
    """Refine phone poses and write the head fit's provenance and clock diagnostics."""
    segment = episode.sync.episode_key.segment
    quest_pose_ms = (episode.sync.offsets_ms or {}).get("quest_pose")
    if quest_pose_ms is None:
        print("HEAD_SKIP", segment, "no Quest clock", flush=True)
        return state
    offsets_ms = dict(episode.sync.offsets_ms or {})
    known = [v for k, v in offsets_ms.items() if k.startswith("exo/") and v is not None]
    guess = float(np.median(known)) if known else quest_pose_ms
    provisional = frozenset(p for p in episode.layout.phones if offsets_ms.get("exo/" + p) is None)
    offsets_ms.update({"exo/" + p: guess for p in provisional})
    track = head_track(episode.hands, quest_pose_ms)
    clocks: dict[str, float] = {}
    for phone in episode.layout.phones:
        value = offsets_ms["exo/" + phone]
        assert value is not None
        clocks[phone] = value
    head = calibrate_with_head(
        state.exo, replace(episode, sync=replace(episode.sync, offsets_ms=offsets_ms)), track, focals,
        clocks, provisional,
    )
    report = HeadReport(
        head.offsets.tolist(), head.offsets_source, track.convention,
        {p: c.camera.parameters.tolist() for p, c in state.exo.phones.items() if c.camera is not None},
        {p: c.camera.parameters.tolist() for p, c in head.exo.phones.items() if c.camera is not None},
        head.clock_corrections_ms, sorted(provisional), head.phones,
    )
    with atomic_write(output / f"{state.session}.head.json") as staged:
        staged.write_text(to_json(report))
    if head.clock_corrections_ms:
        print("HEAD_CLOCK", segment, head.clock_corrections_ms, flush=True)
    print("HEAD_CALIBRATION", segment, {p: (r.method, r.face_median_px) for p, r in head.phones.items()}, flush=True)
    return replace(state, exo=head.exo)


def main(config: Config) -> CalibSession:
    """Read episodes, fit the OAK and exo cameras, optionally refine with head, then publish."""
    session, keys, dataset = open_session(
        config.segment_ids, config.key_file, config.catalog_url, output=config.output_dir, protected=(), dataset=config.dataset
    )
    with session_output(config.output_dir) as output:
        episodes, segments = read_episodes(dataset, keys)
        factories = catalog_io.read_factories(segments[0])
        oak_fit, episodes = fit_oak_mount(session, episodes, segments, factories["ego"], output, config.oak_egohand_assets)
        focals = vp_focals([phone for episode in episodes for phone in episode.layout.phones]) if config.exo_method == "head" else read_table().focals
        state = CalibSession(
            session, calibrate(session, episodes, focals), oak_fit,
            read_quest_block(segments[0], factories["quest"],
                donors=None if config.donor_catalog_url is None else CatalogClient(config.donor_catalog_url).get_dataset(catalog_io.DATASET_NAME)),
            oak_stereo(factories["ego"]),
        )
        if config.exo_method == "head":
            if len(episodes) != 1:
                raise ValueError("head-anchored calibration handles one episode per session")
            state = refine_with_head(state, episodes[0], focals, output)
        for motion in episodes:
            write_calib_layer(output / motion.sync.episode_key.rrd_name, state, motion)
        with atomic_write(output / f"{session}.json") as staged:
            staged.write_text(to_json(state))
        return state
