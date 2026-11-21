"""RoboCap hand-teacher dataset ``robocap-hands``: 60 s clips of the six-camera rig, pseudo-labelled by EgoHand.

Three stages, each a run of ``main``:

- ``cut`` (CPU): plans the clips of each session and writes each clip's ``base`` layer. Every camera's
  ``VideoStream`` samples are copied bit for bit from the session's base rrd, starting at the camera's first
  keyframe at or after the clip start, on two timelines: ``video_time`` (original) and ``frame_index``
  (``round((t - t0) / period)``, shared by the six cameras; a clip ends before the first index some camera
  lacks). The static rig, transforms, pinholes and distortion are copied as they are.
- ``teacher`` (GPU): EgoHand (RT-DETR hand detector + SOMA hands, SelfCap's TensorRT stack) on every
  ``stride``-th ``frame_index`` of every camera, written as the ``hand_teacher`` layer: per labelled image a
  133-point COCO-WholeBody ``Points2D`` (left hand slots 91-111, right 112-132, NaN elsewhere) with
  ``simplecv.KeypointConfidence2D``, and the hand scores as ``Scalars``. Keypoints are in stored-image pixels;
  ``upside_down`` cameras turn 180 degrees before EgoHand and their keypoints turn back (none by default, see ``Config``).
- ``register``: registers the clips' layers on the catalog (additively, replacing a re-written layer).

Labelling rule per image (camera, frame): a candidate box is *confident* at score >= ``hand_threshold`` and
*ambiguous* at ``min_threshold`` <= score < ``hand_threshold``. Hand identity is resolved per image with
upstream's ``resolve_hand_identities`` (at most one left and one right). The image gets no row when an
ambiguous or an unselected confident candidate overlaps no selected hand (IoU < ``duplicate_iou``): such a
candidate is another hand whose presence the teacher cannot decide, while an overlapping one is a duplicate
query of a hand already labelled. It also gets no row when EgoHand returns no pose for a selected hand. Every
other image gets a row, with all-NaN keypoints when it has no hand.
"""

import json
import re
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor

APPLICATION_ID: str = "robocap-hands"
"""Application id of every layer file; the recording id is the segment id."""
BASE_RRDS: Path = Path("/mnt/nas/datasets/robocap/rrd/base")
"""One base rrd per RoboCap session (read only)."""
OUT_DIR: Path = Path("/mnt/nas/ssd-8tb/scratch/hand-models-2026-11-17/robocap-hands")
"""Layer files (``base/``, ``hand_teacher/``), clip plans and status."""
CAMERAS: int = 6
"""Cameras of the RoboCap head rig."""
EVAL_SESSIONS: frozenset[tuple[str, int]] = frozenset(
    [("f408193e6447b3b0", n) for n in (8, 11, 16, 21, 22, 27, 44)] + [("fe6fede545c972fa", n) for n in (13, 15)]
)
"""Sessions held out for evaluation (fixed by the lead); every other session is TRAIN."""
ASTRA_KEYFRAMES: Path = Path("/mnt/nas/ssd-8tb/scratch/hand-labeler/layers/reference_astra_keyframes")
"""Astra-reviewed hand boxes of some EVAL sessions; eval clips prefer windows holding their times."""
SESSION_RE: re.Pattern[str] = re.compile(r"^robocap__([0-9a-f]{16})__s(\d{8})$")
"""Session stem: serial and session number."""
VIDEO_RE: re.Pattern[str] = re.compile(r"^/world/rig_00/cam_(\d{2})/pinhole/video$")
"""Video stream entities of the rig's cameras."""
VIDEO_COLUMNS: tuple[str, ...] = ("video_time", "VideoStream:sample")
"""Columns copied from each video chunk (some base rrds keep ``is_keyframe`` in chunks of its own; the flag is re-attached by time)."""
KEYFRAME_FIELD: pa.Field = (
    Chunk.from_columns("/", indexes=[rr.TimeColumn("video_time", duration=[0])], columns=rr.VideoStream.columns(is_keyframe=[True]))
    .to_record_batch()
    .schema.field("VideoStream:is_keyframe")
)
"""The ``VideoStream:is_keyframe`` column as Rerun writes it (field metadata included)."""
FRAME_INDEX_FIELD: pa.Field = (
    Chunk.from_columns("/", indexes=[rr.TimeColumn("frame_index", sequence=[0])], columns=rr.Scalars.columns(scalars=[0.0])).to_record_batch().schema.field("frame_index")
)
"""The ``frame_index`` sequence timeline column as Rerun writes it (field metadata included)."""
HAND_SLOTS: dict[str, int] = {"left": 91, "right": 112}
"""First COCO-WholeBody slot of each hand (21 slots each, OpenPose order)."""
CLIP_ROWS_PER_CHUNK: int = 30
"""Video rows per written chunk (one second, as in the base rrds)."""


Split = Literal["train", "eval"]


@serde
@dataclass(frozen=True, slots=True)
class ClipPlan:
    """One planned clip of a session: where it starts and which samples each camera keeps."""

    segment: str
    """``robocap__<serial>__s<8 digits>__t<start second, 5 digits>``."""
    session: str
    """Session stem."""
    serial: str
    """Rig serial."""
    split: str
    """``train`` or ``eval``."""
    t0_ns: int
    """Clip origin on the shared ``video_time`` clock: frame_index 0."""
    start_s: float
    """Clip origin in seconds since the session's first common frame."""
    period_ns: float
    """Frame period of the session (median frame spacing)."""
    frames: int
    """frame_index values 0..frames-1 that every camera has."""
    first_ns: list[int]
    """Per camera, the time of its first kept sample (its keyframe at/after the clip start)."""
    last_ns: list[int]
    """Per camera, the time of its last kept sample."""


@serde
@dataclass(frozen=True, slots=True)
class TeacherCounts:
    """What the teacher wrote for one clip."""

    segment: str
    """Segment id."""
    images: int
    """Images (camera, labelled frame) the teacher looked at."""
    with_hand: int
    """Rows with at least one hand."""
    no_hand: int
    """Rows without a hand (all NaN)."""
    skipped_ambiguous: int
    """Images without a row: an undecided hand candidate."""
    skipped_no_pose: int
    """Images without a row: EgoHand returned no pose for a selected hand."""
    hands_left: int
    """Left hands written."""
    hands_right: int
    """Right hands written."""
    seconds: float
    """Wall time of the clip."""


@dataclass(frozen=True, slots=True)
class Config:
    """Build the ``robocap-hands`` clip dataset: ``cut`` base layers, run the EgoHand ``teacher``, ``register`` on the catalog."""

    stage: Literal["cut", "teacher", "register"]
    """Which stage to run."""
    split: Literal["train", "eval", "all"] = "all"
    """Which sessions (cut) or clips (teacher, register) to process."""
    sessions: tuple[str, ...] = ()
    """Session stems to cut; empty cuts every session of ``split``."""
    segments: tuple[str, ...] = ()
    """Segment ids for teacher/register; empty takes every planned clip of ``split``."""
    skip_done: bool = True
    """Skip clips whose output layer already exists."""
    base_rrds: Path = BASE_RRDS
    """Session base rrds."""
    out_dir: Path = OUT_DIR
    """Output root."""
    clip_s: float = 60.0
    """Clip length in seconds."""
    edge_s: float = 20.0
    """Seconds skipped at each end of a session (less for short sessions, down to ``min_edge_s``)."""
    min_edge_s: float = 2.0
    """Smallest edge margin."""
    min_clip_s: float = 20.0
    """Shortest clip a short session may give; shorter sessions give none."""
    train_clips: int = 4
    """Clips per TRAIN session at most."""
    eval_clips: int = 2
    """Clips per EVAL session at most."""
    upside_down: tuple[int, ...] = ()
    """Cameras whose frames turn 180 degrees before EgoHand (keypoints turn back to stored pixels). None by default:
    cameras 2 and 3 look down and their stored frames are already ego-upright (the wearer's body at the bottom; image
    down = rig -y). Turned, EgoHand swapped their sides against cameras 0/1 and put fingers along the forearm, and found
    fewer confident hands (s37 t452: 546/515 vs 885/916 hands on cams 2/3)."""
    stride: int = 3
    """The teacher labels every ``stride``-th frame_index."""
    hand_threshold: float = 0.5
    """Detector score from which a hand counts."""
    min_threshold: float = 0.2
    """Detector score from which a candidate makes its image ambiguous unless it duplicates a selected hand."""
    duplicate_iou: float = 0.5
    """IoU with a selected hand from which a candidate is a duplicate query of it."""
    duplicate_inside: float = 0.7
    """Fraction of a candidate inside a selected hand's box from which it is a duplicate (a part of that hand)."""
    decode_batch: int = 16
    """Frames decoded and run per batch."""
    backbone_batch: int = 2
    """Largest TensorRT batch of the DINOv3 backbone; 2 (the smaller cached engine) and 16-frame decode batches keep the process under 5.5 GB."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog the clips register on."""
    dataset: str = "robocap-hands"
    """Catalog dataset name."""
    layers: tuple[str, ...] = ("base", "hand_teacher")
    """Layers ``register`` registers."""


# ---------------------------------------------------------------- cut


@dataclass(slots=True)
class CameraTimes:
    """One camera's sample times and keyframe flags over a session."""

    times: Int64[ndarray, "n"]
    """Sample times (ns) on ``video_time``."""
    keyframe: Bool[ndarray, "n"]
    """Keyframe flag per sample."""


def split_of(serial: str, number: int) -> Split:
    """``eval`` for the held-out sessions, else ``train``."""
    return "eval" if (serial, number) in EVAL_SESSIONS else "train"


def read_times(path: Path) -> dict[int, CameraTimes]:
    """Every camera's sample times and keyframe flags.

    Times come from the chunks holding ``VideoStream:sample``: some base rrds (Cap B, Cap A from s47 on) keep
    ``is_keyframe`` in chunks of its own that list only the keyframes, so their times are not the frame times.
    """
    times: dict[int, list[ndarray]] = defaultdict(list)
    keyframe_times: dict[int, list[ndarray]] = defaultdict(list)
    flags: dict[int, list[ndarray]] = defaultdict(list)
    # The LazyStore loads chunks on demand from the footer manifest (bounded memory).
    for chunk in RrdReader(path).store().stream().filter(content="/world/rig_00/**", is_static=False):
        match = VIDEO_RE.match(chunk.entity_path)
        if match is None:
            continue
        batch: pa.RecordBatch = chunk.to_record_batch()
        cam: int = int(match[1])
        # Copies: a zero-copy view would pin the chunk's whole buffer, samples included (a 4.3 GB session held 5.6 GB).
        stamps: Int64[ndarray, "n"] = batch.column("video_time").cast(pa.int64()).to_numpy().copy()
        if "VideoStream:sample" in batch.schema.names:
            times[cam].append(stamps)
        if "VideoStream:is_keyframe" in batch.schema.names:
            keyframe_times[cam].append(stamps)
            flags[cam].append(np.asarray([bool(cell[0]) if cell else False for cell in batch.column("VideoStream:is_keyframe").to_pylist()]))
    out: dict[int, CameraTimes] = {}
    for cam in sorted(times):
        frame_times: Int64[ndarray, "n"] = np.unique(np.concatenate(times[cam]))
        flagged: Int64[ndarray, "k"] = np.concatenate(keyframe_times[cam])[np.concatenate(flags[cam])]
        out[cam] = CameraTimes(frame_times, np.isin(frame_times, flagged))
    if sorted(out) != list(range(CAMERAS)):
        raise ValueError(f"{path}: expected cameras 0..{CAMERAS - 1}, found {sorted(out)}")
    return out


def astra_times_ns(session: str) -> Int64[ndarray, "k"]:
    """The times Astra reviewed in a session (any camera, any side); empty when it has no reference file."""
    path: Path = ASTRA_KEYFRAMES / f"{session}.rrd"
    if not path.exists():
        return np.zeros(0, np.int64)
    stamps: list[ndarray] = [
        chunk.to_record_batch().column("video_time").cast(pa.int64()).to_numpy()
        for chunk in RrdReader(path).stream().filter(is_static=False)
        if "video_time" in chunk.timeline_names
    ]
    return np.unique(np.concatenate(stamps)) if stamps else np.zeros(0, np.int64)


def plan_clip(cams: dict[int, CameraTimes], start_ns: int, end_limit_ns: int, period: float, clip_ns: int) -> tuple[int, int, list[int], list[int]] | None:
    """Cut one clip at ``start_ns``: each camera from its first keyframe at/after it, all on one frame_index grid.

    Returns ``(t0, frames, first_ns, last_ns)``, or None when the cameras' keyframes do not line up within half a
    frame or the clip would be shorter than half its length.
    """
    firsts: list[int] = []
    for cam in range(CAMERAS):
        times, keyframe = cams[cam].times, cams[cam].keyframe
        candidates: Int64[ndarray, "k"] = np.flatnonzero(keyframe & (times >= start_ns))
        if not len(candidates):
            return None
        firsts.append(int(times[candidates[0]]))
    if max(firsts) - min(firsts) > period / 2:
        return None
    t0: int = max(firsts)
    wanted: int = int(round(clip_ns / period))
    present: Bool[ndarray, "f"] = np.ones(wanted, dtype=bool)
    for cam in range(CAMERAS):
        times = cams[cam].times
        inside: Int64[ndarray, "m"] = times[(times >= firsts[cam]) & (times < min(end_limit_ns, t0 + clip_ns) - period / 2)]
        index: Int64[ndarray, "m"] = np.round((inside - t0) / period).astype(np.int64)
        # Exactly one sample per index: a missing frame or two samples rounding to one index both end the clip.
        present &= np.bincount(index[(index >= 0) & (index < wanted)], minlength=wanted)[:wanted] == 1
    frames: int = int(np.argmin(present)) if not present.all() else wanted
    if frames < wanted // 2:
        return None
    lasts: list[int] = []
    for cam in range(CAMERAS):
        times = cams[cam].times
        index = np.round((times - t0) / period).astype(np.int64)
        lasts.append(int(times[(times >= firsts[cam]) & (index < frames)].max()))
    return t0, frames, firsts, lasts


def plan_session(config: Config, path: Path) -> tuple[list[ClipPlan], dict[int, CameraTimes]]:
    """Plan a session's clips: up to ``train_clips`` (``eval_clips``) windows, spread evenly or over Astra's times."""
    session: str = path.stem
    match = SESSION_RE.match(session)
    assert match is not None, session
    serial, number = match[1], int(match[2])
    split: Split = split_of(serial, number)
    cams: dict[int, CameraTimes] = read_times(path)
    period: float = float(np.median(np.diff(cams[0].times)))
    begin: int = max(int(c.times[0]) for c in cams.values())
    end: int = min(int(c.times[-1]) for c in cams.values())
    duration_s: float = (end - begin) / 1e9
    clip_s: float = min(config.clip_s, duration_s - 2 * config.min_edge_s - 1.0)
    if clip_s < config.min_clip_s:
        return [], cams
    edge_s: float = max(config.min_edge_s, min(config.edge_s, (duration_s - clip_s - 1.0) / 2))
    clip_ns: int = int(clip_s * 1e9)
    # Recording gaps (a camera skips more than 1.5 frames): no clip crosses one.
    gaps: list[tuple[int, int]] = [
        (int(c.times[i]), int(c.times[i + 1])) for c in cams.values() for i in np.flatnonzero(np.diff(c.times) > 1.5 * period).tolist()
    ]
    lo: int = begin + int(edge_s * 1e9)
    hi: int = end - int(edge_s * 1e9) - clip_ns - int(1.5e9)  # room for the keyframe after the start
    if hi < lo:
        hi = lo
    starts: Int64[ndarray, "g"] = np.arange(lo, hi + 1, int(1e9), dtype=np.int64)
    valid: list[int] = [int(s) for s in starts if not any(g0 < s + clip_ns + 1.5e9 and g1 > s for g0, g1 in gaps)]
    if not valid:
        return [], cams
    wanted: int = config.eval_clips if split == "eval" else config.train_clips
    count: int = max(1, min(wanted, int((hi - lo) / 1e9 // clip_s) + 1))
    if split == "eval":
        astra: Int64[ndarray, "k"] = astra_times_ns(session)
        score: list[int] = [int(np.sum((astra >= s) & (astra < s + clip_ns))) for s in valid]
        targets: list[int] = [valid[i] for i in np.argsort(score, kind="stable")[::-1]] if astra.size else []
        if not targets:
            targets = [int(t) for t in np.linspace(valid[0], valid[-1], count)]
    else:
        targets = [int(t) for t in np.linspace(valid[0], valid[-1], count)]
    plans: list[ClipPlan] = []
    taken: list[tuple[int, int]] = []
    for target in targets:
        if len(plans) >= count:
            break
        for start in sorted(valid, key=lambda s: abs(s - target)):
            if any(start < t1 and start + clip_ns + 1.5e9 > t0_ for t0_, t1 in taken):
                continue
            cut = plan_clip(cams, start, end, period, clip_ns)
            if cut is None:
                continue
            t0, frames, firsts, lasts = cut
            start_s: float = (t0 - begin) / 1e9
            plans.append(
                ClipPlan(
                    segment=f"{session}__t{int(start_s):05d}", session=session, serial=serial, split=split, t0_ns=t0, start_s=start_s,
                    period_ns=period, frames=frames, first_ns=firsts, last_ns=lasts,
                )
            )
            taken.append((start, int(t0 + frames * period)))
            break
    return sorted(plans, key=lambda p: p.t0_ns), cams


def plans_path(config: Config, session: str) -> Path:
    """Where a session's clip plans are kept (JSON written by ``cut``)."""
    return config.out_dir / "plans" / f"{session}.json"


def load_plans(config: Config) -> list[ClipPlan]:
    """Every planned clip of ``config.split`` (and ``config.segments`` when given)."""
    plans: list[ClipPlan] = []
    for path in sorted((config.out_dir / "plans").glob("*.json")):
        for item in json.loads(path.read_text()):
            plan: ClipPlan = from_json(ClipPlan, json.dumps(item))
            if config.split != "all" and plan.split != config.split:
                continue
            if config.segments and plan.segment not in config.segments:
                continue
            plans.append(plan)
    return plans


def video_slice(batch: pa.RecordBatch, plan: ClipPlan, cam: int, keyframes: CameraTimes) -> pa.RecordBatch | None:
    """The rows of one video chunk that clip ``plan`` keeps for camera ``cam``, with their keyframe flag and ``frame_index``."""
    times: Int64[ndarray, "n"] = batch.column("video_time").cast(pa.int64()).to_numpy()
    keep: Bool[ndarray, "n"] = (times >= plan.first_ns[cam]) & (times <= plan.last_ns[cam])
    if not keep.any():
        return None
    # ``take`` copies the kept rows into buffers of their own: a slice would pin the reader's whole read block in memory
    # (cutting a 4.3 GB session held 5.6 GB; the 22 GB sessions ran out of memory).
    rows: pa.RecordBatch = batch.select(list(VIDEO_COLUMNS)).take(pa.array(np.flatnonzero(keep)))
    flags: Bool[ndarray, "m"] = keyframes.keyframe[np.searchsorted(keyframes.times, times[keep])]
    index: Int64[ndarray, "m"] = np.round((times[keep] - plan.t0_ns) / plan.period_ns).astype(np.int64)
    rows = rows.append_column(KEYFRAME_FIELD, pa.array([[bool(f)] for f in flags], KEYFRAME_FIELD.type))
    return rows.append_column(FRAME_INDEX_FIELD, pa.array(index, pa.int64())).replace_schema_metadata(None)


def write_rrd(path: Path, chunks: list[Chunk], segment: str) -> None:
    """Write chunks to ``path`` (via a temporary name, then mode 644)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = path.with_suffix(".partial.rrd")
    LazyChunkStream.from_iter(chunks).write_rrd(partial, application_id=APPLICATION_ID, recording_id=segment)
    partial.chmod(0o644)
    partial.rename(path)


def cut_session(config: Config, path: Path) -> list[ClipPlan]:
    """Plan a session's clips and write each clip's ``base`` layer in one pass over the session's chunks."""
    started: float = time.perf_counter()
    plans, cams = plan_session(config, path)
    out: Path = plans_path(config, path.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_json(plans))
    out.chmod(0o644)
    todo: list[ClipPlan] = [p for p in plans if not (config.skip_done and (config.out_dir / "base" / f"{p.segment}.rrd").exists())]
    if not todo:
        print(f"[cut] {path.stem}: {len(plans)} clips, all written", flush=True)
        return plans
    statics: list[Chunk] = []
    rows: dict[tuple[str, int], list[pa.RecordBatch]] = defaultdict(list)
    for chunk in RrdReader(path).store().stream().filter(content="/world/**"):
        if chunk.is_static:
            statics.append(chunk)
            continue
        match = VIDEO_RE.match(chunk.entity_path)
        if match is None:
            continue
        cam: int = int(match[1])
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "VideoStream:sample" not in batch.schema.names:
            continue  # an is_keyframe-only chunk: the flags come from read_times
        bounds: Int64[ndarray, "n"] = batch.column("video_time").cast(pa.int64()).to_numpy()
        for plan in todo:
            first: int = plan.first_ns[cam]
            last: int = plan.last_ns[cam]
            if bounds[-1] < first or bounds[0] > last:
                continue
            kept: pa.RecordBatch | None = video_slice(batch, plan, cam, cams[cam])
            if kept is not None:
                rows[(plan.segment, cam)].append(kept)
    for plan in todo:
        chunks: list[Chunk] = list(statics)
        for cam in range(CAMERAS):
            table: pa.Table = pa.Table.from_batches(rows[(plan.segment, cam)]).combine_chunks()
            index: Int64[ndarray, "m"] = table.column("frame_index").to_numpy()
            keyframes: list = table.column("VideoStream:is_keyframe").to_pylist()
            if not keyframes or not keyframes[0] or not keyframes[0][0]:
                raise ValueError(f"{plan.segment}: cam {cam} does not start on a keyframe")
            if not np.array_equal(index, np.arange(plan.frames)):
                raise ValueError(f"{plan.segment}: cam {cam} frame_index does not cover 0..{plan.frames - 1}")
            for batch in table.to_batches(max_chunksize=CLIP_ROWS_PER_CHUNK):
                chunks.extend(Chunk.from_record_batch(batch, entity_path=f"/world/rig_00/cam_{cam:02d}/pinhole/video"))
        chunks.append(
            Chunk.from_property(
                "robocap_clip",
                rr.AnyValues(
                    serial=plan.serial, session=plan.session, split=plan.split, clip_start_s=plan.start_s, t0_ns=plan.t0_ns,
                    frame_period_ns=plan.period_ns, num_frames=plan.frames,
                ),
            )
        )
        write_rrd(config.out_dir / "base" / f"{plan.segment}.rrd", chunks, plan.segment)
    print(f"[cut] {path.stem}: {len(plans)} clips ({len(todo)} written) in {time.perf_counter() - started:.1f} s", flush=True)
    return plans


# ---------------------------------------------------------------- teacher


@dataclass(slots=True)
class ClipVideo:
    """One camera of a clip: its encoded samples, their times and frame indices, and its calibration."""

    samples: list[memoryview]
    """Encoded samples in decode order (first is a keyframe)."""
    keyframes: list[bool]
    """Keyframe flag per sample."""
    times_ns: Int64[ndarray, "n"]
    """``video_time`` per sample."""
    frame_index: Int64[ndarray, "n"]
    """``frame_index`` per sample."""
    intrinsics: Float64[ndarray, "3 3"]
    """Pinhole K of the stored image."""
    distortion: Float64[ndarray, "4"]
    """KB4 coefficients k1..k4."""
    width: int
    """Image width."""
    height: int
    """Image height."""


def read_clip(path: Path) -> dict[int, ClipVideo]:
    """Every camera of a clip's base layer."""
    tables: dict[int, list[pa.RecordBatch]] = defaultdict(list)
    calib: dict[int, dict[str, object]] = defaultdict(dict)
    for chunk in RrdReader(path).stream().filter(content="/world/rig_00/**"):
        batch: pa.RecordBatch = chunk.to_record_batch()
        match = VIDEO_RE.match(chunk.entity_path)
        if match is not None and not chunk.is_static:
            tables[int(match[1])].append(batch.select(["video_time", "frame_index", "VideoStream:is_keyframe", "VideoStream:sample"]))
            continue
        pinhole = re.match(r"^/world/rig_00/cam_(\d{2})/pinhole$", chunk.entity_path)
        if pinhole is not None and chunk.is_static:
            for name in batch.schema.names:
                if name in ("Pinhole:image_from_camera", "Pinhole:resolution", "simplecv.components.DistortionModel", "simplecv.components.DistortionCoefficients"):
                    calib[int(pinhole[1])][name] = batch.column(name).to_pylist()[0]
    out: dict[int, ClipVideo] = {}
    for cam in range(CAMERAS):
        table: pa.Table = pa.Table.from_batches([b.replace_schema_metadata(None) for b in tables[cam]]).combine_chunks()
        info: dict[str, object] = calib[cam]
        model = info["simplecv.components.DistortionModel"]
        model_name: str = str(model[0] if isinstance(model, list) else model)
        coefficients: Float64[ndarray, "k"] = np.asarray(info["simplecv.components.DistortionCoefficients"], dtype=np.float64).reshape(-1)
        if model_name != "kannala_brandt" or np.any(coefficients[4:] != 0.0):
            raise ValueError(f"{path}: cam {cam} lens {model_name} {coefficients} is not KB4")
        # Rerun stores Mat3x3 column-major.
        k: Float64[ndarray, "3 3"] = np.asarray(info["Pinhole:image_from_camera"], dtype=np.float64).reshape(-1)[:9].reshape(3, 3).T
        width, height = (int(v) for v in np.asarray(info["Pinhole:resolution"]).reshape(-1)[:2])
        samples_col = table.column("VideoStream:sample").combine_chunks().flatten()  # one sample per row
        data: memoryview = memoryview(samples_col.values.to_numpy(zero_copy_only=False))
        offsets: Int64[ndarray, "n1"] = samples_col.offsets.to_numpy().astype(np.int64)
        out[cam] = ClipVideo(
            samples=[data[offsets[i] : offsets[i + 1]] for i in range(len(samples_col))],
            keyframes=[bool(cell[0]) if cell else False for cell in table.column("VideoStream:is_keyframe").to_pylist()],
            times_ns=table.column("video_time").cast(pa.int64()).to_numpy(),
            frame_index=table.column("frame_index").to_numpy(),
            intrinsics=k,
            distortion=coefficients[:4].copy(),
            width=width,
            height=height,
        )
    return out


def inside_fraction(a: Float32[ndarray, "n 4"], b: Float32[ndarray, "m 4"]) -> Float32[ndarray, "n m"]:
    """Fraction of each XYXY box of ``a`` that lies inside each box of ``b``."""
    lt: ndarray = np.maximum(a[:, None, :2], b[None, :, :2])
    rb: ndarray = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter: ndarray = np.clip(rb - lt, 0.0, None).prod(axis=-1)
    return (inter / np.maximum((a[:, 2:] - a[:, :2]).prod(axis=-1)[:, None], 1e-6)).astype(np.float32)


def box_iou(a: Float32[ndarray, "n 4"], b: Float32[ndarray, "m 4"]) -> Float32[ndarray, "n m"]:
    """Pairwise IoU of XYXY boxes."""
    lt: ndarray = np.maximum(a[:, None, :2], b[None, :, :2])
    rb: ndarray = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter: ndarray = np.clip(rb - lt, 0.0, None).prod(axis=-1)
    area_a: ndarray = (a[:, 2:] - a[:, :2]).prod(axis=-1)
    area_b: ndarray = (b[:, 2:] - b[:, :2]).prod(axis=-1)
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)).astype(np.float32)


@dataclass(slots=True)
class Teacher:
    """The EgoHand detector and hand model, set up once."""

    detector: object
    """``EgoHandDetector`` (raw candidates come from its runtime)."""
    predictor: object
    """``EgoHandPredictor``."""


def setup_teacher(config: Config) -> Teacher:
    """SelfCap's accelerated EgoHand (TRT fp32 detector, TRT bf16 DINOv3; CPU crops), detector threshold at ``min_threshold``."""
    from dataclasses import replace

    from posekit.models.egohand import selfcap_egohand_configs

    detector_config, predictor_config = selfcap_egohand_configs(None)
    detector_config = replace(detector_config, threshold=config.min_threshold, gpu_preprocessing=True)
    # GPU crops take only rational or pinhole lenses; RoboCap's KB4 fisheye crops on the CPU (upstream's reference path).
    backbone = replace(predictor_config.backbone_backend, max_batch_size=config.backbone_batch, opt_batch_size=config.backbone_batch)
    return Teacher(detector_config.setup(), replace(predictor_config, gpu_crops=False, backbone_backend=backbone).setup())


def raw_candidates(teacher: Teacher, frames_rgb: UInt8[Tensor, "b h w 3"]) -> tuple[Float32[ndarray, "n 4"], Float32[ndarray, "n"], Int64[ndarray, "n"], list[str]]:
    """Every detector candidate above the detector threshold (both classes, before identity assignment): boxes, scores, frame, side."""
    from posekit.models.egohand import EgoHandDetector
    from posekit.ops.pil_resize import resize_bilinear_uint8
    from posekit.runtimes import run_chunked

    detector = teacher.detector
    assert isinstance(detector, EgoHandDetector)
    height, width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    boxes: list[Tensor] = []
    scores: list[Tensor] = []
    frames: list[Tensor] = []
    sides: list[str] = []
    step: int = detector.runtime.spec.max_batch_size
    for start in range(0, len(frames_rgb), step):
        chunk: UInt8[Tensor, "c h w 3"] = frames_rgb[start : start + step]
        sizes: Tensor = torch.empty((len(chunk), 2), dtype=torch.int64, device=frames_rgb.device)
        sizes[:, 0] = width
        sizes[:, 1] = height
        outputs: dict[str, Tensor] = run_chunked(detector.runtime, {"images": resize_bilinear_uint8(chunk, (640, 640)).contiguous(), "orig_target_sizes": sizes})
        labels: Tensor = outputs["labels"]
        keep: Tensor = torch.stack([labels == 1, labels == 2], dim=1) & (outputs["scores"] > detector.config.threshold)[:, None]
        hits: tuple[Tensor, ...] = torch.where(keep)
        boxes.append(outputs["boxes"][hits[0], hits[2]].float())
        scores.append(outputs["scores"][hits[0], hits[2]].float())
        frames.append(hits[0] + start)
        sides.extend("left" if label == 0 else "right" for label in hits[1].tolist())  # class 1 left, 2 right
    return (
        torch.cat(boxes).cpu().numpy().astype(np.float32),
        torch.cat(scores).cpu().numpy().astype(np.float32),
        torch.cat(frames).cpu().numpy().astype(np.int64),
        sides,
    )


@dataclass(slots=True)
class ImageLabel:
    """The teacher's verdict on one image."""

    status: Literal["labelled", "ambiguous", "no_pose"]
    """Row (``labelled``) or no row, and why."""
    keypoints: dict[str, Float32[ndarray, "21 2"]] = field(default_factory=dict)
    """Stored-image pixels of each labelled hand."""
    scores: dict[str, float] = field(default_factory=dict)
    """Detector score of each labelled hand."""
    max_ambiguous: float = 0.0
    """Highest undecided candidate score (0 when none)."""


def label_batch(config: Config, teacher: Teacher, frames_rgb: UInt8[Tensor, "b h w 3"], camera: object, flip: bool) -> list[ImageLabel]:
    """Label a batch of (already turned) frames of one camera; keypoints come back in the turned frames' pixels."""
    from posekit.models.egohand import resolve_hand_identities
    from posekit.predictions import SOMA_HAND21, BoxDetections, HandDetections

    device: torch.device = frames_rgb.device
    boxes, scores, frames, sides = raw_candidates(teacher, frames_rgb)
    labels: list[ImageLabel] = []
    selected_boxes: list[ndarray] = []
    selected_scores: list[float] = []
    selected_frames: list[int] = []
    selected_sides: list[str] = []
    for frame in range(len(frames_rgb)):
        rows: Int64[ndarray, "k"] = np.flatnonzero(frames == frame)
        confident: Int64[ndarray, "c"] = rows[scores[rows] >= config.hand_threshold]
        chosen: list[int] = []
        chosen_sides: list[str] = []
        if len(confident):
            single = HandDetections(
                BoxDetections(
                    torch.from_numpy(boxes[confident]).to(device), torch.from_numpy(scores[confident]).to(device),
                    torch.zeros(len(confident), dtype=torch.int64, device=device),
                ),
                tuple(sides[i] for i in confident.tolist()),
            )
            resolved = resolve_hand_identities(single, 1)
            resolved_xyxy: ndarray = resolved.boxes.xyxy_numpy()
            for box, side in zip(resolved_xyxy, resolved.sides, strict=True):
                match: int = int(confident[np.argmin(np.abs(boxes[confident] - box).sum(axis=1))])
                chosen.append(match)
                chosen_sides.append(side)
        others: Int64[ndarray, "o"] = np.asarray([i for i in rows.tolist() if i not in chosen], dtype=np.int64)
        undecided: float = 0.0
        if len(others):
            if chosen:
                picked: ndarray = boxes[np.asarray(chosen)]
                overlap: ndarray = box_iou(boxes[others], picked).max(axis=1)
                lone: ndarray = others[(overlap < config.duplicate_iou) & (inside_fraction(boxes[others], picked).max(axis=1) < config.duplicate_inside)]
            else:
                lone = others
            undecided = float(scores[lone].max()) if len(lone) else 0.0
        chosen_scores: dict[str, float] = {side: float(scores[i]) for i, side in zip(chosen, chosen_sides, strict=True)}
        if undecided > 0.0:
            labels.append(ImageLabel("ambiguous", scores=chosen_scores, max_ambiguous=undecided))
            continue
        labels.append(ImageLabel("labelled", scores=chosen_scores))
        for i, side in zip(chosen, chosen_sides, strict=True):
            selected_boxes.append(boxes[i])
            selected_scores.append(float(scores[i]))
            selected_frames.append(frame)
            selected_sides.append(side)
    if selected_frames:
        # Only frames with a hand go to the hand model (its CPU crops copy every frame it gets to the host).
        with_hands: list[int] = sorted(set(selected_frames))
        local: dict[int, int] = {frame: i for i, frame in enumerate(with_hands)}
        detections = HandDetections(
            BoxDetections(
                torch.from_numpy(np.stack(selected_boxes)).to(device), torch.tensor(selected_scores, dtype=torch.float32, device=device),
                torch.tensor([local[f] for f in selected_frames], dtype=torch.int64, device=device),
            ),
            tuple(selected_sides),  # pyrefly: ignore[bad-argument-type]
        )
        with torch.inference_mode():
            hands = teacher.predictor(frames_rgb[with_hands], detections, camera)  # pyrefly: ignore[not-callable]
        xy: Float32[ndarray, "m 25 2"] = hands.keypoints.xy.cpu().numpy()
        for row, (frame, side) in enumerate(zip(hands.keypoints.frame_indices.tolist(), hands.sides, strict=True)):
            labels[with_hands[frame]].keypoints[side] = xy[row][np.asarray(SOMA_HAND21)].astype(np.float32)
        for frame, label in enumerate(labels):
            if label.status == "labelled" and set(label.scores) != set(label.keypoints):
                labels[frame] = ImageLabel("no_pose", max_ambiguous=label.max_ambiguous)
    return labels


def teacher_clip(config: Config, teacher: Teacher, plan: ClipPlan) -> TeacherCounts:
    """Run the teacher on one clip and write its ``hand_teacher`` layer."""
    from simplecv.catalog_video_codec import wrap_mp4
    from simplecv.rerun_custom_types import Points2DWithConfidence
    from simplecv.video_io import bind_cuda_device
    from torchcodec.decoders import VideoDecoder

    from posekit.cameras import CalibratedCamera

    started: float = time.perf_counter()
    prefetch: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
    videos: dict[int, ClipVideo] = read_clip(config.out_dir / "base" / f"{plan.segment}.rrd")
    device: torch.device = torch.device("cuda")
    chunks: list[Chunk] = []
    counts: dict[str, int] = defaultdict(int)
    for cam in range(CAMERAS):
        video: ClipVideo = videos[cam]
        flip: bool = cam in config.upside_down
        k: Float64[ndarray, "3 3"] = video.intrinsics.copy()
        if flip:
            # A 180 degree turn negates camera x and y: same focal lengths and radial lens, mirrored principal point.
            k[0, 2] = video.width - 1 - k[0, 2]
            k[1, 2] = video.height - 1 - k[1, 2]
        camera = CalibratedCamera(k, video.distortion, video.width, video.height, lens="KB4")
        decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, 30, "h264"), dimension_order="NHWC", device=str(device))
        wanted: Int64[ndarray, "w"] = np.flatnonzero((video.frame_index % config.stride == 0) & (video.frame_index >= 0) & (video.frame_index < plan.frames))
        positions: list[Float32[ndarray, "133 2"]] = []
        confidences: list[Float32[ndarray, "133"]] = []
        row_times: list[int] = []
        row_index: list[int] = []
        score_rows: list[list[float]] = []
        score_times: list[int] = []
        score_index: list[int] = []
        batches: list[Int64[ndarray, "b"]] = [wanted[start : start + config.decode_batch] for start in range(0, len(wanted), config.decode_batch)]

        def decode(picks: Int64[ndarray, "b"], decoder: VideoDecoder = decoder) -> UInt8[Tensor, "b h w 3"]:
            bind_cuda_device(str(device))  # NVDEC needs the CUDA context current on the decoding thread
            return decoder.get_frames_at(picks.tolist()).data.to(device)

        # NVDEC decodes the next batch on a worker thread while this one is labelled.
        pending: Future[Tensor] | None = prefetch.submit(decode, batches[0]) if batches else None
        for number, picks in enumerate(batches):
            assert pending is not None
            frames_rgb: UInt8[Tensor, "b h w 3"] = pending.result()
            pending = prefetch.submit(decode, batches[number + 1]) if number + 1 < len(batches) else None
            if flip:
                frames_rgb = torch.flip(frames_rgb, dims=(1, 2)).contiguous()
            for pick, label in zip(picks.tolist(), label_batch(config, teacher, frames_rgb, camera, flip), strict=True):
                counts["images"] += 1
                score_rows.append([label.scores.get("left", 0.0), label.scores.get("right", 0.0), label.max_ambiguous])
                score_times.append(int(video.times_ns[pick]))
                score_index.append(int(video.frame_index[pick]))
                if label.status != "labelled":
                    counts[f"skipped_{label.status}"] += 1
                    continue
                coco: Float32[ndarray, "133 2"] = np.full((133, 2), np.nan, dtype=np.float32)
                conf: Float32[ndarray, "133"] = np.zeros(133, dtype=np.float32)
                for side, uv in label.keypoints.items():
                    if flip:
                        uv = np.stack([video.width - 1 - uv[:, 0], video.height - 1 - uv[:, 1]], axis=1)
                    first: int = HAND_SLOTS[side]
                    coco[first : first + 21] = uv
                    inside: Bool[ndarray, "21"] = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < video.width) & (uv[:, 1] >= 0) & (uv[:, 1] < video.height)
                    conf[first : first + 21] = np.where(inside, label.scores[side], 0.0)
                    counts[f"hands_{side}"] += 1
                counts["with_hand" if label.keypoints else "no_hand"] += 1
                positions.append(coco)
                confidences.append(conf)
                row_times.append(int(video.times_ns[pick]))
                row_index.append(int(video.frame_index[pick]))
        entity: str = f"/world/rig_00/cam_{cam:02d}/pinhole/hand_teacher"
        if positions:
            indexes = [rr.TimeColumn("video_time", duration=np.asarray(row_times, dtype="timedelta64[ns]")), rr.TimeColumn("frame_index", sequence=row_index)]
            columns = Points2DWithConfidence.columns(positions=np.concatenate(positions), confidences=np.concatenate(confidences)).partition([133] * len(positions))
            chunks.append(Chunk.from_columns(entity, indexes=indexes, columns=columns))
        chunks.append(
            Chunk.from_columns(
                f"{entity}/scores",
                indexes=[rr.TimeColumn("video_time", duration=np.asarray(score_times, dtype="timedelta64[ns]")), rr.TimeColumn("frame_index", sequence=score_index)],
                columns=rr.Scalars.columns(scalars=np.asarray(score_rows, dtype=np.float64)),
            )
        )
    chunks.append(
        Chunk.from_property(
            "hand_teacher",
            rr.AnyValues(
                model="EgoHand (nvidia RT-DETR hand detector + EgoHand SOMA hands), SelfCap TensorRT stack",
                stride=config.stride, upside_down=list(config.upside_down), hand_threshold=config.hand_threshold, min_threshold=config.min_threshold, duplicate_iou=config.duplicate_iou,
                layout="COCO-WholeBody 133: left hand 91-111, right hand 112-132, OpenPose hand order; stored-image pixels",
                scores="scores entity per looked-at image: [left score, right score, highest undecided candidate score]; 0 when absent",
            ),
        )
    )
    prefetch.shutdown()
    write_rrd(config.out_dir / "hand_teacher" / f"{plan.segment}.rrd", chunks, plan.segment)
    result = TeacherCounts(
        segment=plan.segment, images=counts["images"], with_hand=counts["with_hand"], no_hand=counts["no_hand"],
        skipped_ambiguous=counts["skipped_ambiguous"], skipped_no_pose=counts["skipped_no_pose"],
        hands_left=counts["hands_left"], hands_right=counts["hands_right"], seconds=time.perf_counter() - started,
    )
    stats: Path = config.out_dir / "teacher_counts" / f"{plan.segment}.json"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(to_json(result))
    stats.chmod(0o644)
    print(f"[teacher] {result}", flush=True)
    return result


# ---------------------------------------------------------------- register


def register(config: Config, plans: list[ClipPlan]) -> None:
    """Register each layer file of ``plans`` on the catalog dataset (replacing a segment's layer if it was re-written)."""
    from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer

    entry = CatalogClient(config.catalog_url).create_dataset(config.dataset, exist_ok=True)
    for layer in config.layers:
        paths: list[Path] = [config.out_dir / layer / f"{plan.segment}.rrd" for plan in plans]
        uris: list[str] = [path.resolve().as_uri() for path in paths if path.exists()]
        if uris:
            entry.register(uris, layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        print(f"[register] {config.dataset} layer {layer}: {len(uris)} of {len(paths)} segments", flush=True)


def main(config: Config) -> None:
    """Run one stage of the ``robocap-hands`` build."""
    if config.stage == "cut":
        paths: list[Path] = sorted(config.base_rrds.glob("robocap__*.rrd"))
        if config.sessions:
            paths = [config.base_rrds / f"{session}.rrd" for session in config.sessions]  # in the given order
        for path in paths:
            match = SESSION_RE.match(path.stem)
            if match is None or (config.split != "all" and split_of(match[1], int(match[2])) != config.split):
                continue
            cut_session(config, path)
        return
    plans: list[ClipPlan] = load_plans(config)
    if config.stage == "register":
        register(config, plans)
        return
    teacher: Teacher = setup_teacher(config)
    for plan in plans:
        if config.skip_done and (config.out_dir / "hand_teacher" / f"{plan.segment}.rrd").exists():
            continue
        teacher_clip(config, teacher, plan)
