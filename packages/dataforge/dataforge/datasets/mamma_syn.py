"""MammaSyn, MAMMA's synthetic training set, rebuilt from its WebDataset shards into the MPI families' layers.

``--family syn`` converts MAMMA's own renders: one recording per sequence directory, every camera's PNG frames encoded to an
AV1 video, the parameters the render was made from as body_pose, and its instance masks. ``--family syn-bedlam`` (the BEDLAM
release repacked) is the other family. The shards are read in place and never modified. See ``docs/mamma.md``.
"""

import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Generic, TypeVar

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray

from dataforge import blueprints, hands, paths, schema, transports, writing
from dataforge.clocks import frame_times
from dataforge.datasets.base import DataforgeDataset
from dataforge.datasets.mamma import MammaConfig, layout, scene_view, select_sequences
from dataforge.datasets.mamma_body import PosedPerson, SmplxBody, fetch_models
from dataforge.datasets.mamma_layers import (
    MASK_DOWNSCALE,
    MASK_WORKERS,
    PERSON_COLOURS,
    SOURCE_TAGS,
    CameraVisibility,
    Clock,
    decode_gray,
    log_cameras,
    log_person_masks,
    palette_png,
    stored_pixels,
    stored_size,
    vertex_rows,
    write_body_mesh,
    write_body_pose,
    write_projections,
    write_visibility,
)
from dataforge.datasets.mamma_source import VERTICES, PersonFit, SmplxModel
from dataforge.datasets.mamma_syn_download import MAMMASYN_REPO, MAMMASYN_REVISION, fetch_proof, verify_syn
from dataforge.datasets.mamma_syn_source import (
    BEDLAM_DATASET,
    BEDLAM_NUMBERING_FPS,
    BEDLAM_STRIDE,
    RENDER_FPS,
    SYN_DIR,
    BedlamClip,
    BedlamLabels,
    BedlamPersonLabel,
    BedlamView,
    PersonContact,
    RenderLabels,
    RenderPersonLabel,
    RenderView,
    ShardMember,
    ShardReader,
    SynFamily,
    SynMetadata,
    SynSample,
    SynSequence,
    assemble_bedlam,
    assemble_render,
    discover_bedlam,
    discover_renders,
    read_bedlam_view,
    read_members,
    read_render_view,
    sequence_samples,
    stage_shards,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    annotation_context,
    frame_index_column,
    layout_frustum_m,
    log_camera_source,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.records import read_json
from dataforge.timing import SequenceTimer
from dataforge.transports import HfFileInfo
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, encode_frames_to_mp4, work_dir

JOINT_TOLERANCE_M: float = 1e-4
"""Largest gap allowed between the shipped joints3d and the same joints posed from the shipped parameters with the person's
model. Measured on the release: under 0.001 mm with the right model, 0.58 mm or more with the other one."""
MODEL_CANDIDATES: tuple[SmplxModel, ...] = ("locked_head_neutral", "v1_1_neutral")
"""The neutral models MammaSyn's bodies were rendered with: the locked head (Interhand) or v1.1 (Inter-X, BEDLAM, ...); the
shards do not say which, so each person takes the one whose forward gives back the shipped joints."""
MESH_STRIDE: int = 1
"""Every frame keeps its mesh: at 6 fps a coarser stride would leave the body still for half a second."""
SYN_LAYERS: tuple[str, ...] = (
    paths.BASE_LAYER,
    paths.BODY_POSE_LAYER,
    paths.BODY_MESH_LAYER,
    paths.PROJECTIONS_LAYER,
    paths.MASKS_LAYER,
    paths.CONTACTS_LAYER,
    paths.VISIBILITY_LAYER,
)
PIXEL_LAYERS: frozenset[str] = frozenset({paths.BASE_LAYER, paths.MASKS_LAYER})
"""The layers that need a sample's PNGs; the others need only its labels and the first PNG's size."""
PNG_HEADER_BYTES: int = 24
"""The bytes of a PNG that hold its width and height."""
CONTACT_THRESHOLD_M: float = 0.01
"""A vertex nearer the other person's surface than this, or inside it, is in contact with them."""
FLOOR_CONTACT_COLOUR: tuple[int, int, int] = (255, 140, 0)
PARTNER_CONTACT_COLOUR: tuple[int, int, int] = (230, 50, 200)
CONTACT_RADIUS_M: float = 0.006
VISIBILITY_SOURCE: str = "each sample's label pickle: vertex_visibility, person_visitiblity_rate (the release's spelling)"
SYN_RIGS: int = 8
"""Most cameras a MammaSyn render sequence has; the family layout's scene rules cover them all."""
VIDEO_SOURCE: str = f"AV1 NVENC encode (CQ {AV1_CQ}, 4:2:0) of the shards' RGBA PNG renders, alpha dropped"


ViewT = TypeVar("ViewT", RenderView, BedlamView)


@dataclass
class CameraRead(Generic[ViewT]):
    """One camera's pass over its samples: its labels, its masks and its image size."""

    camera: str
    frames: list[int] = field(default_factory=list)
    views: list[ViewT] = field(default_factory=list)
    masks: list[bytes] = field(default_factory=list)
    size: tuple[int, int] | None = None
    """The shipped PNGs' width and height."""
    turns: int = 0
    """Clockwise quarter turns from the shipped PNGs and masks to the stored video and masks."""

    @property
    def turned_size(self) -> tuple[int, int]:
        """The camera image's width and height: the PNGs' after the turns."""
        assert self.size is not None  # a camera with samples has its PNG size
        width, height = self.size
        return (height, width) if self.turns % 2 else (width, height)


def parse_render(sample: SynSample, payload: bytes, origin: str) -> RenderView:
    return RenderView(sample.camera, sample.frame, read_render_view(payload, origin))


def parse_bedlam(sample: SynSample, payload: bytes, origin: str) -> BedlamView:
    return BedlamView(sample.frame, read_bedlam_view(payload, origin))


def png_size(payload: bytes, origin: str) -> tuple[int, int]:
    """Width and height from a PNG's header."""
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{origin}: not a PNG")
    return int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")


def stream_camera(  # noqa: UP047 — beartype requires legacy generics
    directory: Path,
    samples: list[SynSample],
    read: CameraRead[ViewT],
    parse: Callable[[SynSample, bytes, str], ViewT],
    members: dict[tuple[str, int], bytes] | None = None,
    *,
    pixels: bool = True,
) -> Iterator[bytes]:
    """Read one camera's samples in frame order (from ``members`` when already read): yield each render's PNG, keep its labels
    and mask on ``read``. Without ``pixels``, keep only the labels and the size from the first PNG's header (``label_members``)."""
    with ShardReader(directory, members) as shards:
        for sample in samples:
            origin: str = f"{directory}/{sample.camera}/{sample.frame:04d}"
            image: bytes = shards.read(sample.image) if pixels else b""
            if pixels or read.size is None:
                size: tuple[int, int] = png_size(image if pixels else shards.read(png_header(sample)), origin)
                if read.size not in (None, size):
                    raise ValueError(f"{origin}: {size[0]}x{size[1]}, the camera's earlier frames are {read.size}")
                read.size = size
            read.frames.append(sample.frame)
            read.views.append(parse(sample, shards.read(sample.labels), origin))
            if pixels:
                read.masks.append(shards.read(sample.mask))
                yield image


def png_header(sample: SynSample) -> ShardMember:
    """The bytes of a sample's PNG that hold its size."""
    return replace(sample.image, size=PNG_HEADER_BYTES)


def label_members(samples: list[SynSample]) -> list[ShardMember]:
    """What a pass without pixels reads of ``samples`` (one camera's, in frame order): every label pickle and the first PNG's header."""
    return [png_header(samples[0]), *(sample.labels for sample in samples)]


def read_cameras(  # noqa: UP047 — beartype requires legacy generics
    directory: Path,
    by_camera: dict[str, list[SynSample]],
    work: Path | None,
    parse: Callable[[SynSample, bytes, str], ViewT],
    members: dict[tuple[str, int], bytes] | None = None,
    turns: int = 0,
    *,
    pixels: bool = True,
) -> list[CameraRead[ViewT]]:
    """Every camera's pass, in parallel; with ``work``, each camera's renders are encoded to ``work/<camera>.mp4`` on the way,
    turned ``turns`` quarters clockwise. Without ``pixels`` (and so without ``work``), the labels alone (``stream_camera``)."""

    def run(camera: str) -> CameraRead[ViewT]:
        read: CameraRead[ViewT] = CameraRead(camera, turns=turns)
        frames: Iterator[bytes] = stream_camera(directory, by_camera[camera], read, parse, members, pixels=pixels)
        if work is None:
            for _ in frames:
                pass
        else:
            # The nominal container rate; log_video_stream stamps every sample with its source frame's time.
            encoded: int = encode_frames_to_mp4(
                frames, work / f"{camera}.mp4", source=FrameSource("png"), fps=RENDER_FPS, gop=AV1_GOP, cq=AV1_CQ, rotate_cw_quarter_turns=turns
            )
            if encoded != len(by_camera[camera]):
                raise ValueError(f"{directory}/{camera}: {encoded} frames encoded of {len(by_camera[camera])}")
        return read

    with ThreadPoolExecutor(max_workers=len(by_camera)) as executor:
        return list(executor.map(run, sorted(by_camera)))


def fit_model(body: SmplxBody, person: PersonFit, joints3d: Float64[ndarray, "t 127 3"], cam_T_body: Float64[ndarray, "t 4 4"] | None = None) -> PersonFit:
    """``person`` with the candidate model whose forward is closest to the shipped joints at the first row it is present
    (moved into the camera frame first when ``cam_T_body`` is given, as BEDLAM's joints are)."""
    assert person.present is not None and person.present.any()  # MammaSyn fits always carry their presence
    row: Int64[ndarray, "1"] = np.flatnonzero(person.present)[:1]
    gaps: dict[SmplxModel, float] = {}
    for model in MODEL_CANDIDATES:
        posed: PosedPerson = body.pose(replace(person, model=model), row, mesh_stride=1)
        if cam_T_body is not None:
            posed = in_camera(posed, cam_T_body[row])
        gaps[model] = float(np.abs(posed.joints[0, :127] - joints3d[row[0]]).max())
    return replace(person, model=min(gaps, key=lambda model: gaps[model]))


def check_joints(
    people: list[PersonFit], people_posed: list[PosedPerson], joints3d: list[Float64[ndarray, "t 127 3"]], frames: Int64[ndarray, "t"]
) -> float:
    """The largest gap between each person's shipped joints3d and the posed ones over the rows the person is present at.

    Raises:
        ValueError: A present row holds a non-finite joint (a broken label or forward; absent rows are NaN by design), or a gap
            exceeds ``JOINT_TOLERANCE_M``; names the person and frame.
    """
    worst: float = 0.0
    for person, posed, shipped in zip(people, people_posed, joints3d, strict=True):
        rows: Int64[ndarray, "p"] = np.flatnonzero(person.present)
        gaps: Float64[ndarray, "p"] = np.abs(posed.joints[rows, :127] - shipped[rows]).max(axis=(1, 2))
        if not np.isfinite(gaps).all():
            raise ValueError(f"{person.subject}: frame {frames[rows[np.flatnonzero(~np.isfinite(gaps))[0]]]} has a non-finite shipped or posed joint")
        row: int = int(gaps.argmax())
        if gaps[row] > JOINT_TOLERANCE_M:
            raise ValueError(f"{person.subject}: the shipped parameters miss the shipped joints3d by {gaps[row] * 1000:.1f} mm at frame {frames[rows[row]]}")
        worst = max(worst, float(gaps[row]))
    return worst


def pose_people(
    body: SmplxBody,
    people: list[PersonFit],
    joints3d: list[Float64[ndarray, "t 127 3"]],
    clock: Clock,
    cam_T_body: list[Float64[ndarray, "t 4 4"]] | None = None,
) -> tuple[list[PersonFit], list[PosedPerson], float]:
    """Each person with the model that gives back their shipped joints, posed over the clock's rows (moved into the camera frame
    by ``cam_T_body``, as BEDLAM's are), and the largest joint gap (``check_joints``)."""
    rows: Int64[ndarray, "t"] = clock.rows
    transforms: list[Float64[ndarray, "t 4 4"]] | list[None] = [None] * len(people) if cam_T_body is None else cam_T_body
    fitted: list[PersonFit] = [fit_model(body, person, joints, moved) for person, joints, moved in zip(people, joints3d, transforms, strict=True)]
    posed: list[PosedPerson] = [body.pose(person, rows, mesh_stride=MESH_STRIDE) for person in fitted]
    posed = [pose if moved is None else in_camera(pose, moved) for pose, moved in zip(posed, transforms, strict=True)]
    return fitted, posed, check_joints(fitted, posed, joints3d, clock.frames)


def camera_clock(read: CameraRead[RenderView] | CameraRead[BedlamView], fps: int) -> Clock:
    """The camera's own samples on the sequence clock (a camera may lack a frame another has)."""
    frames: Int64[ndarray, "n"] = np.array(read.frames, dtype=np.int64)
    return Clock(frames, frame_times(frames, fps))


def write_render_base(
    recording: rr.RecordingStream,
    identity: SequenceIdentity,
    sequence: SynSequence,
    labels: RenderLabels,
    reads: list[CameraRead[RenderView]],
    work: Path,
    *,
    shard_samples: int,
    joint_error_m: float,
    timer: SequenceTimer,
) -> None:
    """Cameras (fixed pinholes; a render has no lens) and every camera's video, on the source frame numbers at 6 fps."""
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True, recording=recording)  # the renders' world: y down, as BEDLAM's
    rr.log("/", annotation_context(), static=True, recording=recording)
    floor_xz: Float64[ndarray, "c 2"] = np.array([camera.world_T_cam[[0, 2], 3] for camera in labels.cameras])
    resolutions, _ = log_cameras(recording, labels.cameras, layout_frustum_m(floor_xz))
    for rig, read in enumerate(reads, start=1):
        clock: Clock = camera_clock(read, RENDER_FPS)
        with timer.stage("write:video"):
            log_video_stream(recording, work / f"{read.camera}.mp4", schema.video_path(rig, 0), times_ns=clock.times_ns, frame_indices=clock.frames)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(labels.frames),
        num_cameras=len(labels.cameras),
        source_num_frames=len(labels.frames),
        fps=RENDER_FPS,
        clock_source=f"source frame number / {RENDER_FPS} fps (MAMMA renders MammaSyn at {RENDER_FPS} fps); a frame missing from the "
        "shards is a gap on the clock",
        source_resolution=pa.array(resolutions),
        video_source=VIDEO_SOURCE,
    )
    recording.send_property(
        "episode",
        rr.AnyValues(
            group=sequence.group,
            collection=sequence.dataset,
            sequence=sequence.directory.name,
            body_source=SOURCE_TAGS["mamma_render"],
            fit_method="mamma_render",
            num_people=len(labels.people),
            subjects=pa.array([person.subject for person in labels.people]),
            shard_samples=shard_samples,
            joints3d_max_error_mm=joint_error_m * 1000,
        ),
    )


def instance_mask_pngs(source: bytes, size: tuple[int, int], turns: int) -> dict[int, bytes]:
    """A palette PNG per person present in a sample's instance mask (pixel value = person index + 1), keyed by person index;
    the mask turned ``turns`` quarters clockwise first, as its video."""
    ids = np.rot90(decode_gray(source), k=-turns)  # decoded once for every person; no jaxtyping on thread-pool locals (see palette_png)
    return {
        int(value) - 1: palette_png(np.where(ids == value, np.uint8(255), np.uint8(0)), size, PERSON_COLOURS[(int(value) - 1) % len(PERSON_COLOURS)])
        for value in np.unique(ids)
        if value
    }


def write_instance_masks(recording: rr.RecordingStream, reads: list[CameraRead[RenderView]] | list[CameraRead[BedlamView]], people: list[int], fps: int) -> None:
    """Every camera's instance masks split per person: everyone the labels name, and anyone else a mask shows (a sample's labels
    can leave out a person its mask has). Palette PNGs at a quarter size; an empty one where the person is not in the mask."""
    recording.send_property(
        paths.MASKS_LAYER,
        rr.AnyValues(
            encoding=f"palette PNG at 1/{MASK_DOWNSCALE} of the stored video's width and height, scaled over it by a Transform3D: "
            "index 1 = the person, index 0 = background (transparent)",
            person_ids="person_NN = the shards' instance mask value NN + 1",
        ),
    )
    stored: list[tuple[int, int]] = [stored_size(*read.turned_size) for read in reads]
    sizes: list[tuple[int, int]] = [(width // MASK_DOWNSCALE, height // MASK_DOWNSCALE) for width, height in stored]
    cv2.setNumThreads(1)  # see MASK_WORKERS
    with ThreadPoolExecutor(max_workers=MASK_WORKERS) as executor:
        futures: list[list[Future[dict[int, bytes]]]] = [
            [executor.submit(instance_mask_pngs, mask, size, read.turns) for mask in read.masks] for read, size in zip(reads, sizes, strict=True)
        ]
        split: list[list[dict[int, bytes]]] = [[future.result() for future in camera] for camera in futures]
    everyone: list[int] = sorted(set(people).union(*(frame for camera in split for frame in camera)))
    for rig, (read, frames, size) in enumerate(zip(reads, split, sizes, strict=True), start=1):
        clock: Clock = camera_clock(read, fps)
        for person in everyone:
            empty: bytes = palette_png(np.zeros(size[::-1], dtype=np.uint8), size, PERSON_COLOURS[person % len(PERSON_COLOURS)])
            blobs: list[bytes] = [frame.get(person, empty) for frame in frames]
            log_person_masks(recording, rig, person, stored[rig - 1], size, blobs, [time_column(clock.times_ns), frame_index_column(clock.frames)])


def log_contact(
    recording: rr.RecordingStream,
    path: str,
    pose: PosedPerson,
    mesh_of: Int64[ndarray, "t"],
    kept: Bool[ndarray, "t"],
    in_contact: Bool[ndarray, "t 10475"],
    clock: Clock,
    *,
    colour: tuple[int, int, int],
    statics: rr.AnyValues,
    shipped: dict[str, pa.Array] | None = None,
) -> None:
    """One contact entity: the shipped columns and ``in_contact`` at the ``kept`` rows (empty elsewhere), and the contacting
    vertices of the posed body drawn as points. ``mesh_of`` gives each row's posed mesh in ``pose.vertices``; every kept row has
    one."""
    rr.log(path, statics, static=True, recording=recording)
    columns = rr.DynamicArchetype.columns(archetype="dataforge.VertexContact", components={**(shipped or {}), "in_contact": vertex_rows(in_contact, kept)})
    rr.send_columns(path, indexes=clock.indexes, columns=columns, recording=recording)
    rows, vertices = np.nonzero(in_contact & kept[:, None])  # row-major, so the points come row by row
    rr.log(path, rr.Points3D.from_fields(colors=colour, radii=CONTACT_RADIUS_M), static=True, recording=recording)
    rr.send_columns(
        path,
        indexes=clock.indexes,
        columns=rr.Points3D.columns(positions=pose.vertices[mesh_of[rows], vertices]).partition(np.where(kept, in_contact.sum(axis=1), 0)),
        recording=recording,
    )


def write_contacts(recording: rr.RecordingStream, posed: list[PosedPerson], contacts: list[PersonContact], clock: Clock) -> None:
    """Every person's floor contact and signed distance to the other person, one row per frame (an empty one where the person is
    absent or the frame ships no distance), the contacting vertices drawn on the posed body."""
    recording.send_property(
        paths.CONTACTS_LAYER,
        rr.AnyValues(
            source="each sample's label pickle: floor_contact_mask, sdf_vertices (the same in every view of a frame)",
            contact_threshold_m=CONTACT_THRESHOLD_M,
            partner_contact="in_contact = signed_distance < contact_threshold_m: inside the other person counts as contact",
        ),
    )
    for pose, contact in zip(posed, contacts, strict=True):
        mesh_of: Int64[ndarray, "t"] = np.full(len(clock.frames), -1, dtype=np.int64)
        mesh_of[pose.mesh_rows[pose.mesh_present]] = np.arange(len(pose.vertices))
        present: Bool[ndarray, "t"] = mesh_of >= 0  # every row the person is at (MESH_STRIDE keeps a mesh on each)
        log_contact(
            recording,
            schema.person_floor_contact_path(pose.index),
            pose,
            mesh_of,
            present,
            contact.floor,
            clock,
            colour=FLOOR_CONTACT_COLOUR,
            statics=rr.AnyValues(n_vertices=VERTICES, topology="smplx"),
        )
        if contact.partner is None or contact.partner_distance_m is None:
            continue
        shipped: Bool[ndarray, "t"] = present & np.isfinite(contact.partner_distance_m).all(axis=1)
        log_contact(
            recording,
            schema.person_contact_path(pose.index, contact.partner),
            pose,
            mesh_of,
            shipped,
            np.nan_to_num(contact.partner_distance_m, nan=np.inf) < CONTACT_THRESHOLD_M,  # on the shipped float32 distance
            clock,
            colour=PARTNER_CONTACT_COLOUR,
            statics=rr.AnyValues(
                n_vertices=VERTICES, topology="smplx", contact_threshold_m=CONTACT_THRESHOLD_M, unit="metres, negative inside the other person"
            ),
            shipped={"signed_distance": vertex_rows(contact.partner_distance_m.astype(np.float16), shipped)},
        )


def camera_visibility(
    rig: int, person: int, views: list[RenderView] | list[BedlamView], frames: Int64[ndarray, "t"], landmarks_uv: Float32[ndarray, "t l 2"]
) -> CameraVisibility:
    """One camera's visibility of one person over the sequence's ``frames``, from the camera's views that label them, with the
    landmarks' pixels."""
    row_of: dict[int, int] = {int(frame): row for row, frame in enumerate(frames)}
    labelled: Bool[ndarray, "t"] = np.zeros(len(frames), dtype=bool)
    visible: Bool[ndarray, "t 10475"] = np.zeros((len(frames), VERTICES), dtype=bool)
    rate: Float32[ndarray, "t"] = np.full(len(frames), np.nan, dtype=np.float32)
    for view in views:
        label: RenderPersonLabel | BedlamPersonLabel | None = view.people.get(person)
        if label is not None:
            row: int = row_of[view.frame]
            labelled[row], visible[row], rate[row] = True, label.vertex_visibility, label.person_visitiblity_rate
    return CameraVisibility(rig, person, labelled, visible, None, rate, landmarks_uv)


def render_visibility(reads: list[CameraRead[RenderView]], labels: RenderLabels, posed: list[PosedPerson]) -> list[CameraVisibility]:
    """What each camera sees of each person, from the camera's own views; the landmarks through its pinhole, at the stored size."""
    by_camera: dict[str, CameraRead[RenderView]] = {read.camera: read for read in reads}
    return [
        camera_visibility(rig, pose.index, by_camera[camera.name].views, labels.frames, stored_pixels(camera, pose.landmarks))
        for rig, camera in enumerate(labels.cameras, start=1)
        for pose in posed
    ]


def syn_download(config: MammaConfig, family: SynFamily, selected: Callable[[], list[tuple[Path, list[str]]]]) -> None:
    """Fetch the SMPL-X models and verify the family's shards in place against the pinned Hub listing; with ``fetch_into``,
    fetch the shards holding the chosen sequences (``selected``: each one's directory and shard names) and compare them."""
    fetched: int = fetch_models(config.model_root)
    listing: list[HfFileInfo] = transports.hf_list_files(MAMMASYN_REPO, MAMMASYN_REVISION)
    files, problems = verify_syn(config.root, family, listing)
    print(f"{config.name} download: {files} shard and metadata files verified in place under {config.root / SYN_DIR}; {fetched} SMPL-X file(s) fetched")
    if config.fetch_into is not None:
        if config.sequences is None:
            raise ValueError("--fetch-into proves the Hub download on chosen sequences: name them with --sequences (MammaSyn is 6.63 TB)")
        paths.require_outside([config.fetch_into], roots=[paths.NAS_ROOT, config.root])
        for directory, shards in selected():
            mismatches: list[str] = fetch_proof(config.root, directory, shards, config.fetch_into)
            problems += mismatches
            print(f"  fetched {len(shards)} shard(s) of {directory.name} from the Hub: {mismatches or 'identical to the release copy'}")
    for problem in problems:
        print(f"  problem: {problem}")
    if problems:
        raise ValueError(f"{config.name}: {len(problems)} problem(s), listed above")


class MammaSynDataset(DataforgeDataset[MammaConfig, SynSequence]):
    """Convert one MammaSyn sequence at a time from the shards in place."""

    layers = SYN_LAYERS

    @cached_property
    def body(self) -> SmplxBody:
        return SmplxBody(self.config.model_root)

    @cached_property
    def stage_root(self) -> Path:
        """This process's directory for the local shard copies, beside the rrd tree (``work/mamma-syn-stage/<pid>``): a preview
        and a full run never share one. The directories of processes that died are removed first."""
        parent: Path = paths.work_root() / "mamma-syn-stage"
        parent.mkdir(parents=True, exist_ok=True)
        for directory in parent.iterdir():
            if directory.name.isdigit():
                try:
                    os.kill(int(directory.name), 0)
                except ProcessLookupError:
                    shutil.rmtree(directory, ignore_errors=True)
                except PermissionError:
                    pass  # another user's live process
        root: Path = parent / str(os.getpid())
        root.mkdir(exist_ok=True)
        return root

    def download(self) -> None:
        syn_download(self.config, "syn", lambda: [(sequence.directory, [shard.name for shard in sequence.shards]) for _, sequence in self.discover()])

    def discover(self) -> list[tuple[SequenceIdentity, SynSequence]]:
        found, skipped = discover_renders(self.config.root)
        for directory, reason in skipped:
            print(f"  mamma: skipped {directory}: {reason}")
        return select_sequences(self.config, [(SequenceIdentity(self.config.name, sequence.parts), sequence) for sequence in found], "sequence")

    def prefetch(self, identity: SequenceIdentity, source: SynSequence, *, force: bool) -> None:
        """Copy the shards of a sequence whose pixels are still to convert to local disk, and index the copy, while the one before
        it converts. A sequence that needs its labels alone reads them in place (``convert``)."""
        if PIXEL_LAYERS.intersection(self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])[1]):
            staged: Path = self.stage_root / identity.recording_id
            try:
                sequence_samples(stage_shards(source.shards, staged), self.config.index_root, identity.sequence_key)
            except BaseException:
                shutil.rmtree(staged, ignore_errors=True)  # a failed prefetch fails the sequence before its convert could remove the copy
                raise

    def convert(self, identity: SequenceIdentity, source: SynSequence, *, force: bool) -> Path:
        """Read every camera once (encoding its renders when base is pending), assemble and check the labels, write the layers.

        A sequence whose pending layers need no pixels reads its label pickles in place, never staging its shards.
        """
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        pixels: bool = bool(PIXEL_LAYERS.intersection(pending))
        staged: Path = self.stage_root / identity.recording_id
        with work_dir("mamma-syn-") as work:
            try:
                with self.timer.stage("read"):
                    # The prefetch's copies, unless this is the run's first sequence; the shards in place for labels alone.
                    copies: list[Path] = stage_shards(source.shards, staged) if pixels else source.shards
                    samples: list[SynSample] = sequence_samples(copies, self.config.index_root, identity.sequence_key)
                    shipped: SynMetadata = read_json(source.metadata, SynMetadata)
                if shipped.num_samples != len(samples):
                    raise ValueError(f"{source.directory}: {len(samples)} samples in the shards, metadata.json declares {shipped.num_samples}")
                frames: list[int] = sorted({sample.frame for sample in samples})
                kept: set[int] = set(frames[: self.config.frame_limit or len(frames)])
                by_camera: dict[str, list[SynSample]] = defaultdict(list)
                for sample in samples:
                    if sample.frame in kept:
                        by_camera[sample.camera].append(sample)
                with self.timer.stage("transcode" if paths.BASE_LAYER in pending else "read"):
                    reads: list[CameraRead[RenderView]] = read_cameras(
                        staged if pixels else source.directory, by_camera, work if paths.BASE_LAYER in pending else None, parse_render, pixels=pixels
                    )
            finally:
                shutil.rmtree(staged, ignore_errors=True)
            labels: RenderLabels = assemble_render([view for read in reads for view in read.views], {read.camera: read.size for read in reads if read.size})
            clock: Clock = Clock(labels.frames, frame_times(labels.frames, RENDER_FPS))
            with self.timer.stage("smplx"):
                people, posed, joint_error_m = pose_people(self.body, labels.people, labels.joints3d, clock)
                labels = replace(labels, people=people)
            if paths.BASE_LAYER in pending:
                with (
                    self.timer.stage(f"write:{paths.BASE_LAYER}"),
                    writing.atomic_recording(
                        targets[paths.BASE_LAYER],
                        recording_id=identity.recording_id,
                        default_blueprint=layout(len(labels.cameras), [camera.name for camera in labels.cameras]),
                        send_properties=True,
                    ) as recording,
                ):
                    write_render_base(
                        recording, identity, source, labels, reads, work, shard_samples=len(samples), joint_error_m=joint_error_m, timer=self.timer
                    )
        self.write_layers(
            identity,
            targets,
            [layer for layer in pending if layer != paths.BASE_LAYER],
            {
                paths.BODY_POSE_LAYER: lambda recording: write_body_pose(recording, labels.people, "mamma_render", posed, clock),
                paths.BODY_MESH_LAYER: lambda recording: write_body_mesh(recording, posed, clock),
                paths.PROJECTIONS_LAYER: lambda recording: write_projections(
                    recording, labels.cameras, posed, clock, camera_model="pinhole", calibration_source="each sample's label pickle (cam_int, cam_ext)"
                ),
                paths.MASKS_LAYER: lambda recording: write_instance_masks(recording, reads, [person.index for person in labels.people], RENDER_FPS),
                paths.CONTACTS_LAYER: lambda recording: write_contacts(recording, posed, labels.contacts, clock),
                paths.VISIBILITY_LAYER: lambda recording: write_visibility(
                    recording,
                    render_visibility(reads, labels, posed),
                    self.body.landmarks,
                    clock,
                    source=VISIBILITY_SOURCE,
                ),
            },
        )
        self.timer.capture_s = len(labels.frames) / RENDER_FPS
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return layout(SYN_RIGS, panes=SYN_RIGS)

    def table_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_table_blueprint(scene_view(SYN_RIGS, card=True), blueprints.camera_view("Camera 1", 1, 0))

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("group", "collection", "num_people")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)


BEDLAM_FRUSTUM_M: float = 0.3
"""Frustum length of the BEDLAM camera, which sits at the origin of its own frame with the bodies a few metres ahead."""


def in_camera(posed: PosedPerson, cam_T_body: Float64[ndarray, "t 4 4"]) -> PosedPerson:
    """A person posed in their own world, moved into the camera frame row by row (BEDLAM gives each body its own world)."""

    def moved(points: Float32[ndarray, "r n 3"], rows: Int64[ndarray, "r"]) -> Float32[ndarray, "r n 3"]:
        transforms: Float64[ndarray, "r 4 4"] = cam_T_body[rows]
        return (np.einsum("rij,rnj->rni", transforms[:, :3, :3], points) + transforms[:, None, :3, 3]).astype(np.float32)

    rows: Int64[ndarray, "t"] = np.arange(len(cam_T_body), dtype=np.int64)
    return PosedPerson(
        index=posed.index,
        coco133=moved(posed.coco133, rows),
        joints=moved(posed.joints, rows),
        mesh_rows=posed.mesh_rows,
        mesh_present=posed.mesh_present,
        vertices=moved(posed.vertices, posed.mesh_rows[posed.mesh_present]),
        faces=posed.faces,
        landmarks=moved(posed.landmarks, rows),
    )


def write_bedlam_base(
    recording: rr.RecordingStream,
    identity: SequenceIdentity,
    clip: BedlamClip,
    labels: BedlamLabels,
    read: CameraRead[BedlamView],
    work: Path,
    clock: Clock,
    *,
    joint_error_m: float,
) -> None:
    """The camera at the origin of its own frame (the scene's world), its intrinsics at every frame (a zoom changes them), its video."""
    width, height = read.turned_size
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True, recording=recording)  # the camera frame: x right, y down, z ahead
    rr.log("/", annotation_context(), static=True, recording=recording)
    log_rig_node(recording, 1, reference=None, num_cameras=1, name="camera", kind="ego")
    rr.log(schema.cam_path(1, 0), rr.AnyValues(name="camera", kind="rgb", camera_model="pinhole"), static=True, recording=recording)
    rr.log(schema.cam_path(1, 0), rr.Transform3D(translation=[0.0, 0.0, 0.0]), static=True, recording=recording)  # the world is this camera
    # K changes every frame on a zoom clip, so it is temporal; the resolution rides in the same rows (Rerun 0.38.1 reports an
    # "invalid pinhole resolution" for a temporal K beside a static resolution). A static K would hide the per-frame one.
    rr.log(
        schema.pinhole_path(1, 0),
        rr.Pinhole.from_fields(camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=BEDLAM_FRUSTUM_M),
        static=True,
        recording=recording,
    )
    rr.send_columns(
        schema.pinhole_path(1, 0),
        indexes=clock.indexes,
        columns=rr.Pinhole.columns(image_from_camera=labels.K, resolution=np.tile([float(width), float(height)], (len(labels.K), 1))),
        recording=recording,
    )
    resolution: str = log_camera_source(
        recording, 1, 0, name="camera", source_width=width, source_height=height, video_codec="av1", cq=AV1_CQ, gop=AV1_GOP
    )
    log_video_stream(recording, work / f"{read.camera}.mp4", schema.video_path(1, 0), times_ns=clock.times_ns, frame_indices=clock.frames)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(clock.frames),
        num_cameras=1,
        source_num_frames=len(clock.frames),
        fps=RENDER_FPS,
        clock_source=f"source frame number / {BEDLAM_NUMBERING_FPS} fps (BEDLAM's render rate; the shards hold every {BEDLAM_STRIDE}th frame)",
        source_resolution=pa.array([resolution]),
        video_source=VIDEO_SOURCE,
    )
    recording.send_property(
        "episode",
        rr.AnyValues(
            group="Singles",
            collection=BEDLAM_DATASET.split("/")[-1],
            batch=clip.directory.name,
            sequence=clip.clip,
            body_source=SOURCE_TAGS["bedlam_render"],
            fit_method="bedlam_render",
            num_people=len(labels.people),
            subjects=pa.array([person.subject for person in labels.people]),
            genders=pa.array([person.gender for person in labels.people]),
            shard_samples=len(clip.samples),
            joints3d_max_error_mm=joint_error_m * 1000,
        ),
    )


def write_bedlam_pose(recording: rr.RecordingStream, labels: BedlamLabels, posed: list[PosedPerson], clock: Clock) -> None:
    """The shipped parameters (each person's own world) with each person's world-to-camera transform and the motion they come
    from, and the camera-frame keypoints."""
    write_body_pose(recording, labels.people, "bedlam_render", posed, clock)
    for person, motion in zip(labels.people, labels.motions, strict=True):
        rr.log(
            schema.person_path(person.index),
            rr.AnyValues(motion_action=motion[0], motion_source=motion[2], motion_info=motion),
            static=True,
            recording=recording,
        )
    for person, transforms in zip(labels.people, labels.cam_T_body, strict=True):
        rr.send_columns(
            schema.person_body_path(person.index, "smplx"),
            indexes=clock.indexes,
            columns=rr.AnyValues.columns(cam_T_body=pa.FixedSizeListArray.from_arrays(pa.array(transforms.reshape(-1)), 16)),
            recording=recording,
        )


def frame_pixels(K: Float64[ndarray, "t 3 3"], points: Float32[ndarray, "t n 3"]) -> Float32[ndarray, "t n 2"]:
    """Camera-frame points through each row's K, in pixels; NaN behind the camera."""
    pixels: Float64[ndarray, "t n 3"] = np.einsum("tij,tnj->tni", K, points.astype(np.float64))
    depth: Float64[ndarray, "t n 1"] = pixels[..., 2:]
    return (pixels[..., :2] / np.where(depth > 0, depth, np.nan)).astype(np.float32)


def write_bedlam_projections(recording: rr.RecordingStream, labels: BedlamLabels, posed: list[PosedPerson], clock: Clock) -> None:
    """Every person's camera-frame keypoints through the camera's K at each frame."""
    recording.send_property(paths.PROJECTIONS_LAYER, rr.AnyValues(derived_from="coco133_xyz", camera_model="pinhole", calibration_source="each frame's label cam_int"))
    for pose in posed:
        hands.log_keypoints2d(
            recording,
            schema.person_projection_path(1, 0, pose.index),
            times_ns=clock.times_ns,
            frame_indices=clock.frames,
            positions=frame_pixels(labels.K, pose.coco133),
            confidence=None,
        )


def bedlam_visibility(read: CameraRead[BedlamView], labels: BedlamLabels, posed: list[PosedPerson]) -> list[CameraVisibility]:
    """What the clip's camera sees of each person; the landmarks of the camera-frame body through each frame's K."""
    return [camera_visibility(1, pose.index, read.views, labels.frames, frame_pixels(labels.K, pose.landmarks)) for pose in posed]


class MammaBedlamDataset(DataforgeDataset[MammaConfig, BedlamClip]):
    """Convert one BEDLAM clip at a time from the shards in place, in its camera frame."""

    layers = SYN_LAYERS

    def __init__(self, config: MammaConfig) -> None:
        super().__init__(config)
        self.prefetched: dict[str, dict[tuple[str, int], bytes]] = {}
        """Each prefetched clip's members by recording id, until its convert takes them."""

    @cached_property
    def body(self) -> SmplxBody:
        return SmplxBody(self.config.model_root)

    def download(self) -> None:
        syn_download(
            self.config,
            "syn-bedlam",
            lambda: [(clip.directory, sorted({member.shard for sample in clip.samples for member in (sample.image, sample.mask, sample.labels)})) for _, clip in self.discover()],
        )

    def discover(self) -> list[tuple[SequenceIdentity, BedlamClip]]:
        batches: set[str] | None = None if self.config.sequences is None else {key.split("/")[0] for key in self.config.sequences}
        clips, problems = discover_bedlam(self.config.root, self.config.index_root, batches)
        if problems:
            raise ValueError(f"{self.config.name}: {len(problems)} batch(es) whose shards do not hold what metadata.json declares: {problems}")
        return select_sequences(self.config, [(SequenceIdentity(self.config.name, clip.parts), clip) for clip in clips], "clip")

    def clip_samples(self, clip: BedlamClip) -> list[SynSample]:
        return clip.samples[: self.config.frame_limit or len(clip.samples)]

    def prefetch(self, identity: SequenceIdentity, source: BedlamClip, *, force: bool) -> None:
        """Read the members of a clip still to convert into memory while the clip before it converts (a clip's samples lie in a
        few runs of its batch's shards: a few reads, instead of three random NAS reads per sample on the convert's path). A clip
        whose pending layers need no pixels reads its label pickles and one PNG header (``label_members``)."""
        pending: list[str] = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])[1]
        if pending:
            samples: list[SynSample] = self.clip_samples(source)
            if PIXEL_LAYERS.intersection(pending):
                members: dict[tuple[str, int], bytes] = read_members(source.directory, [member for sample in samples for member in (sample.image, sample.mask, sample.labels)])
            else:
                members = read_members(source.directory, label_members(samples), gap=0)
            self.prefetched[identity.recording_id] = members

    def convert(self, identity: SequenceIdentity, source: BedlamClip, *, force: bool) -> Path:
        """Read the clip once (encoding it when base is pending), pose every person and move them into the camera frame, write."""
        members: dict[tuple[str, int], bytes] | None = self.prefetched.pop(identity.recording_id, None)
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        samples: list[SynSample] = self.clip_samples(source)
        with work_dir("mamma-bedlam-") as work:
            with self.timer.stage("transcode" if paths.BASE_LAYER in pending else "read"):
                (read,) = read_cameras(
                    source.directory,
                    {source.clip: samples},
                    work if paths.BASE_LAYER in pending else None,
                    parse_bedlam,
                    members,
                    source.turns,
                    pixels=bool(PIXEL_LAYERS.intersection(pending)),
                )
            labels: BedlamLabels = assemble_bedlam(read.views)
            # BEDLAM's principal point is the image centre; anywhere else, the PNGs are turned against K (a closeup the batch name missed).
            width, height = read.turned_size
            worst: int = int(np.abs(labels.K[:, :2, 2] - [width / 2, height / 2]).max(axis=1).argmax())
            cx, cy = float(labels.K[worst, 0, 2]), float(labels.K[worst, 1, 2])
            if max(abs(cx - width / 2), abs(cy - height / 2)) > 0.5:
                raise ValueError(f"{source.directory}/{source.clip}: principal point ({cx}, {cy}) is not the centre of the {width}x{height} image")
            clock: Clock = Clock(labels.frames, frame_times(labels.frames, BEDLAM_NUMBERING_FPS))
            with self.timer.stage("smplx"):
                people, posed, joint_error_m = pose_people(self.body, labels.people, labels.joints3d, clock, labels.cam_T_body)
                labels = replace(labels, people=people)
            if paths.BASE_LAYER in pending:
                with (
                    self.timer.stage(f"write:{paths.BASE_LAYER}"),
                    writing.atomic_recording(targets[paths.BASE_LAYER], recording_id=identity.recording_id, default_blueprint=layout(1, ["camera"]), send_properties=True) as recording,
                ):
                    write_bedlam_base(recording, identity, source, labels, read, work, clock, joint_error_m=joint_error_m)
        self.write_layers(
            identity,
            targets,
            [layer for layer in pending if layer != paths.BASE_LAYER],
            {
                paths.BODY_POSE_LAYER: lambda recording: write_bedlam_pose(recording, labels, posed, clock),
                paths.BODY_MESH_LAYER: lambda recording: write_body_mesh(recording, posed, clock),
                paths.PROJECTIONS_LAYER: lambda recording: write_bedlam_projections(recording, labels, posed, clock),
                paths.MASKS_LAYER: lambda recording: write_instance_masks(recording, [read], [person.index for person in labels.people], BEDLAM_NUMBERING_FPS),
                paths.CONTACTS_LAYER: lambda recording: write_contacts(recording, posed, labels.contacts, clock),
                paths.VISIBILITY_LAYER: lambda recording: write_visibility(
                    recording,
                    bedlam_visibility(read, labels, posed),
                    self.body.landmarks,
                    clock,
                    source=VISIBILITY_SOURCE,
                ),
            },
        )
        self.timer.capture_s = (int(labels.frames[-1]) - int(labels.frames[0]) + BEDLAM_STRIDE) / BEDLAM_NUMBERING_FPS
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return layout(1, ["camera"])

    def table_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_table_blueprint(scene_view(1, card=True), blueprints.camera_view("Camera", 1, 0))

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("batch", "num_people")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
