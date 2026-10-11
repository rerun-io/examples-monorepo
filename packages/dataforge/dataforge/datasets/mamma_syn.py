"""MammaSyn, MAMMA's synthetic training set, rebuilt from its WebDataset shards into the MPI families' layers.

``--family syn`` converts MAMMA's own renders: one recording per sequence directory, every camera's PNG frames encoded to an
AV1 video, the parameters the render was made from as body_pose, and its instance masks. ``--family syn-bedlam`` (the BEDLAM
release repacked) is the other family. The shards are read in place and never modified. See ``docs/mamma.md``.
"""

from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray

from dataforge import blueprints, hands, paths, schema, writing
from dataforge.clocks import frame_times
from dataforge.datasets.base import DataforgeDataset
from dataforge.datasets.mamma import FAMILY_PANES, FAMILY_RIGS, MammaConfig, layout, scene_view
from dataforge.datasets.mamma_body import PosedPerson, SmplxBody
from dataforge.datasets.mamma_layers import (
    MASK_DOWNSCALE,
    MASK_WORKERS,
    PERSON_COLOURS,
    Clock,
    decode_gray,
    log_cameras,
    log_person_masks,
    palette_png,
    stored_size,
    write_body_mesh,
    write_body_pose,
    write_projections,
)
from dataforge.datasets.mamma_source import MammaCamera, PersonFit, SequenceLabels
from dataforge.datasets.mamma_syn_source import (
    BEDLAM_DATASET,
    BEDLAM_NUMBERING_FPS,
    RENDER_FPS,
    BedlamClip,
    BedlamLabels,
    BedlamView,
    RenderLabels,
    RenderView,
    ShardReader,
    SynMetadata,
    SynSample,
    SynSequence,
    assemble_bedlam,
    assemble_render,
    discover_bedlam,
    discover_renders,
    read_bedlam_view,
    read_metadata,
    read_render_view,
    sequence_samples,
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
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, encode_frames_to_mp4, work_dir

JOINT_TOLERANCE_M: float = 1e-3
"""Largest gap allowed between the shipped joints3d and the same joints posed from the shipped parameters. Measured on the
release: 0.00 mm (the renders were made from exactly these parameters with the locked-head neutral)."""
MESH_STRIDE: int = 1
"""Every frame keeps its mesh: at 6 fps a coarser stride would leave the body still for half a second."""


ViewT = TypeVar("ViewT", RenderView, BedlamView)


@dataclass
class CameraRead(Generic[ViewT]):
    """One camera's pass over its samples: its labels, its masks and its image size."""

    camera: str
    frames: list[int] = field(default_factory=list)
    views: list[ViewT] = field(default_factory=list)
    masks: list[bytes] = field(default_factory=list)
    size: tuple[int, int] | None = None


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
    directory: Path, samples: list[SynSample], read: CameraRead[ViewT], parse: Callable[[SynSample, bytes, str], ViewT]
) -> Iterator[bytes]:
    """Read one camera's samples in frame order: yield each render's PNG, keep its labels and mask on ``read``."""
    with ShardReader(directory) as shards:
        for sample in samples:
            origin: str = f"{directory}/{sample.camera}/{sample.frame:04d}"
            image: bytes = shards.read(sample.image)
            size: tuple[int, int] = png_size(image, origin)
            if read.size not in (None, size):
                raise ValueError(f"{origin}: {size[0]}x{size[1]}, the camera's earlier frames are {read.size}")
            read.size = size
            read.frames.append(sample.frame)
            read.masks.append(shards.read(sample.mask))
            read.views.append(parse(sample, shards.read(sample.labels), origin))
            yield image


def read_cameras(  # noqa: UP047 — beartype requires legacy generics
    directory: Path, by_camera: dict[str, list[SynSample]], work: Path | None, parse: Callable[[SynSample, bytes, str], ViewT]
) -> list[CameraRead[ViewT]]:
    """Every camera's pass, in parallel; with ``work``, each camera's renders are encoded to ``work/<camera>.mp4`` on the way."""

    def run(camera: str) -> CameraRead[ViewT]:
        read: CameraRead[ViewT] = CameraRead(camera)
        frames: Iterator[bytes] = stream_camera(directory, by_camera[camera], read, parse)
        if work is None:
            for _ in frames:
                pass
        else:
            # The nominal container rate; log_video_stream stamps every sample with its source frame's time.
            encoded: int = encode_frames_to_mp4(frames, work / f"{camera}.mp4", source=FrameSource("png"), fps=RENDER_FPS, gop=AV1_GOP, cq=AV1_CQ)
            if encoded != len(by_camera[camera]):
                raise ValueError(f"{directory}/{camera}: {encoded} frames encoded of {len(by_camera[camera])}")
        return read

    with ThreadPoolExecutor(max_workers=len(by_camera)) as executor:
        return list(executor.map(run, sorted(by_camera)))


def person_ids(people: list[PersonFit]) -> list[int]:
    """Each person's index in the release (``person_NN``): the entity it is logged under and its instance mask value - 1."""
    return [int(person.subject.removeprefix("person_")) for person in people]


def check_joints(people: list[PersonFit], people_posed: list[PosedPerson], joints3d: list[Float64[ndarray, "t 127 3"]]) -> float:
    """The largest gap between each person's shipped joints3d and the posed ones; raises above ``JOINT_TOLERANCE_M``."""
    worst: float = 0.0
    for person, posed, shipped in zip(people, people_posed, joints3d, strict=True):
        gap: float = float(np.nanmax(np.abs(posed.joints[:, :127] - shipped))) if np.isfinite(shipped).any() else 0.0
        if gap > JOINT_TOLERANCE_M:
            raise ValueError(f"{person.subject}: the shipped parameters miss the shipped joints3d by {gap * 1000:.1f} mm")
        worst = max(worst, gap)
    return worst


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
        video_source=f"AV1 NVENC encode (CQ {AV1_CQ}, 4:2:0) of the shards' RGBA PNG renders, alpha dropped",
    )
    recording.send_property(
        "episode",
        rr.AnyValues(
            group=sequence.group,
            collection=sequence.dataset,
            sequence=sequence.directory.name,
            body_source="synthetic_gt",
            fit_method="mamma_render",
            num_people=len(labels.people),
            subjects=pa.array([person.subject for person in labels.people]),
            shard_samples=shard_samples,
            joints3d_max_error_mm=joint_error_m * 1000,
        ),
    )


def instance_mask_pngs(source: bytes, size: tuple[int, int]) -> dict[int, bytes]:
    """A palette PNG per person present in a sample's instance mask (pixel value = person index + 1), keyed by person index."""
    ids = decode_gray(source)  # decoded once for every person; no jaxtyping on thread-pool locals (see palette_png)
    return {
        int(value) - 1: palette_png(np.where(ids == value, 255, 0).astype(np.uint8), size, PERSON_COLOURS[(int(value) - 1) % len(PERSON_COLOURS)])
        for value in np.unique(ids)
        if value
    }


def write_instance_masks(
    recording: rr.RecordingStream, cameras: list[MammaCamera], reads: list[CameraRead[RenderView]] | list[CameraRead[BedlamView]], people: list[int], fps: int
) -> None:
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
    with ThreadPoolExecutor(max_workers=MASK_WORKERS) as executor:
        split: list[list[dict[int, bytes]]] = []
        for camera, read in zip(cameras, reads, strict=True):
            stored: tuple[int, int] = stored_size(camera.width, camera.height)
            split.append(list(executor.map(partial(instance_mask_pngs, size=(stored[0] // MASK_DOWNSCALE, stored[1] // MASK_DOWNSCALE)), read.masks)))
    everyone: list[int] = sorted(set(people).union(*(frame for camera in split for frame in camera)))
    for rig, (camera, read, frames) in enumerate(zip(cameras, reads, split, strict=True), start=1):
        clock: Clock = camera_clock(read, fps)
        stored = stored_size(camera.width, camera.height)
        size: tuple[int, int] = (stored[0] // MASK_DOWNSCALE, stored[1] // MASK_DOWNSCALE)
        for person in everyone:
            empty: bytes = palette_png(np.zeros(size[::-1], dtype=np.uint8), size, PERSON_COLOURS[person % len(PERSON_COLOURS)])
            blobs: list[bytes] = [frame.get(person, empty) for frame in frames]
            log_person_masks(recording, rig, person, stored, size, blobs, [time_column(clock.times_ns), frame_index_column(clock.frames)])


class MammaSynDataset(DataforgeDataset[MammaConfig, SynSequence]):
    """Convert one MammaSyn sequence at a time from the shards in place."""

    layers = (paths.BASE_LAYER, paths.BODY_POSE_LAYER, paths.BODY_MESH_LAYER, paths.PROJECTIONS_LAYER, paths.MASKS_LAYER)

    def __init__(self, config: MammaConfig) -> None:
        super().__init__(config)
        self.body: SmplxBody | None = None

    def download(self) -> None:
        raise NotImplementedError("mamma syn download: not written yet")

    def discover(self) -> list[tuple[SequenceIdentity, SynSequence]]:
        found, skipped = discover_renders(self.config.root)
        for directory, reason in skipped:
            print(f"  mamma: skipped {directory}: {reason}")
        pairs: list[tuple[SequenceIdentity, SynSequence]] = [(SequenceIdentity(self.config.name, sequence.parts), sequence) for sequence in found]
        if self.config.sequences is not None:
            known: set[str] = {identity.sequence_key for identity, _ in pairs}
            missing: set[str] = set(self.config.sequences) - known
            if missing:
                raise ValueError(f"{self.config.name}: no sequence {sorted(missing)} under {self.config.root}")
            pairs = [(identity, sequence) for identity, sequence in pairs if identity.sequence_key in self.config.sequences]
        return pairs

    def convert(self, identity: SequenceIdentity, source: SynSequence, *, force: bool) -> Path:
        """Read every camera once (encoding its renders when base is pending), assemble and check the labels, write the layers."""
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        with self.timer.stage("read"):
            samples: list[SynSample] = sequence_samples(source.directory, self.config.index_root, identity.sequence_key)
            shipped: SynMetadata = read_metadata(source.metadata)
        if shipped.num_samples != len(samples):
            raise ValueError(f"{source.directory}: {len(samples)} samples in the shards, metadata.json declares {shipped.num_samples}")
        frames: list[int] = sorted({sample.frame for sample in samples})
        kept: set[int] = set(frames[: self.config.frame_limit or len(frames)])
        by_camera: dict[str, list[SynSample]] = defaultdict(list)
        for sample in samples:
            if sample.frame in kept:
                by_camera[sample.camera].append(sample)
        with work_dir("mamma-syn-") as work:
            with self.timer.stage("transcode" if paths.BASE_LAYER in pending else "read"):
                reads: list[CameraRead[RenderView]] = read_cameras(source.directory, by_camera, work if paths.BASE_LAYER in pending else None, parse_render)
            labels: RenderLabels = assemble_render([view for read in reads for view in read.views], {read.camera: read.size for read in reads if read.size})
            clock: Clock = Clock(labels.frames, frame_times(labels.frames, RENDER_FPS))
            if self.body is None:
                self.body = SmplxBody(self.config.model_root)
            body: SmplxBody = self.body
            with self.timer.stage("smplx"):
                posed: list[PosedPerson] = [body.pose(person, clock.rows, mesh_stride=MESH_STRIDE) for person in labels.people]
                joint_error_m: float = check_joints(labels.people, posed, labels.joints3d)
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
        sequence_labels: SequenceLabels = SequenceLabels(
            name=source.directory.name, fps=RENDER_FPS, frames=len(labels.frames), method="mamma_render", people=labels.people
        )
        ids: list[int] = person_ids(labels.people)
        self.write_layers(
            identity,
            targets,
            [layer for layer in pending if layer != paths.BASE_LAYER],
            {
                paths.BODY_POSE_LAYER: lambda recording: write_body_pose(recording, sequence_labels, posed, clock, ids),
                paths.BODY_MESH_LAYER: lambda recording: write_body_mesh(recording, posed, clock, ids),
                paths.PROJECTIONS_LAYER: lambda recording: write_projections(
                    recording, labels.cameras, posed, clock, calibration_source="each sample's label pickle (cam_int, cam_ext)", ids=ids
                ),
                paths.MASKS_LAYER: lambda recording: write_instance_masks(recording, labels.cameras, reads, ids, RENDER_FPS),
            },
        )
        self.timer.capture_s = len(labels.frames) / RENDER_FPS
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return layout(FAMILY_RIGS[self.config.family], panes=FAMILY_PANES[self.config.family])

    def table_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_table_blueprint(scene_view(FAMILY_RIGS[self.config.family], card=True), blueprints.camera_view("Camera 1", 1, 0))

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
        coco133=moved(posed.coco133, rows),
        joints=moved(posed.joints, rows),
        mesh_rows=posed.mesh_rows,
        mesh_present=posed.mesh_present,
        vertices=moved(posed.vertices, posed.mesh_rows[posed.mesh_present]),
        faces=posed.faces,
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
    assert read.size is not None
    width, height = read.size
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True, recording=recording)  # the camera frame: x right, y down, z ahead
    rr.log("/", annotation_context(), static=True, recording=recording)
    log_rig_node(recording, 1, reference=None, num_cameras=1, name="camera", kind="ego")
    rr.log(schema.cam_path(1, 0), rr.AnyValues(name="camera", kind="rgb", camera_model="pinhole"), static=True, recording=recording)
    rr.log(schema.cam_path(1, 0), rr.Transform3D(translation=np.zeros(3), mat3x3=np.eye(3), from_parent=True), static=True, recording=recording)
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
    log_video_stream(recording, work / f"{clip.clip}.mp4", schema.video_path(1, 0), times_ns=clock.times_ns, frame_indices=clock.frames)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(clock.frames),
        num_cameras=1,
        source_num_frames=len(clock.frames),
        fps=BEDLAM_NUMBERING_FPS,
        clock_source=f"source frame number / {BEDLAM_NUMBERING_FPS} fps (BEDLAM's render rate; the shards hold every 5th frame)",
        source_resolution=pa.array([resolution]),
        video_source=f"AV1 NVENC encode (CQ {AV1_CQ}, 4:2:0) of the shards' RGBA PNG renders, alpha dropped",
    )
    recording.send_property(
        "episode",
        rr.AnyValues(
            group="Singles",
            collection=BEDLAM_DATASET.split("/")[-1],
            batch=clip.directory.name,
            sequence=clip.clip,
            body_source="synthetic_gt",
            fit_method="bedlam_render",
            num_people=len(labels.people),
            subjects=pa.array([person.subject for person in labels.people]),
            genders=pa.array([person.gender for person in labels.people]),
            shard_samples=len(clip.samples),
            joints3d_max_error_mm=joint_error_m * 1000,
        ),
    )


def write_bedlam_pose(recording: rr.RecordingStream, labels: BedlamLabels, posed: list[PosedPerson], clock: Clock) -> None:
    """The shipped parameters (each person's own world) with each person's world-to-camera transform, and the camera-frame keypoints."""
    sequence_labels: SequenceLabels = SequenceLabels(name="clip", fps=BEDLAM_NUMBERING_FPS, frames=len(clock.frames), method="bedlam_render", people=labels.people)
    write_body_pose(recording, sequence_labels, posed, clock, labels.ids)
    for index, transforms in zip(labels.ids, labels.cam_T_body, strict=True):
        rr.send_columns(
            schema.person_body_path(index, "smplx"),
            indexes=clock.indexes,
            columns=rr.AnyValues.columns(cam_T_body=pa.FixedSizeListArray.from_arrays(pa.array(transforms.reshape(-1)), 16)),
            recording=recording,
        )


def write_bedlam_projections(recording: rr.RecordingStream, labels: BedlamLabels, posed: list[PosedPerson], clock: Clock) -> None:
    """Every person's camera-frame keypoints through the camera's K at each frame."""
    recording.send_property(paths.PROJECTIONS_LAYER, rr.AnyValues(derived_from="coco133_xyz", camera_model="pinhole", calibration_source="each frame's label cam_int"))
    for index, pose in zip(labels.ids, posed, strict=True):
        pixels: Float64[ndarray, "t 133 3"] = np.einsum("tij,tnj->tni", labels.K, pose.coco133.astype(np.float64))
        depth: Float64[ndarray, "t 133 1"] = pixels[..., 2:]
        hands.log_keypoints2d(
            recording,
            schema.person_projection_path(1, 0, index),
            times_ns=clock.times_ns,
            frame_indices=clock.frames,
            positions=(pixels[..., :2] / np.where(depth > 0, depth, np.nan)).astype(np.float32),
            confidence=None,
        )


class MammaBedlamDataset(DataforgeDataset[MammaConfig, BedlamClip]):
    """Convert one BEDLAM clip at a time from the shards in place, in its camera frame."""

    layers = (paths.BASE_LAYER, paths.BODY_POSE_LAYER, paths.BODY_MESH_LAYER, paths.PROJECTIONS_LAYER, paths.MASKS_LAYER)

    def __init__(self, config: MammaConfig) -> None:
        super().__init__(config)
        self.body: SmplxBody | None = None

    def download(self) -> None:
        raise NotImplementedError("mamma syn-bedlam download: not written yet")

    def discover(self) -> list[tuple[SequenceIdentity, BedlamClip]]:
        batches: set[str] | None = None if self.config.sequences is None else {key.split("/")[0] for key in self.config.sequences}
        clips, problems = discover_bedlam(self.config.root, self.config.index_root, batches)
        for problem in problems:
            print(f"  mamma: skipped {problem}")
        pairs: list[tuple[SequenceIdentity, BedlamClip]] = [(SequenceIdentity(self.config.name, clip.parts), clip) for clip in clips]
        if self.config.sequences is not None:
            known: set[str] = {identity.sequence_key for identity, _ in pairs}
            missing: set[str] = set(self.config.sequences) - known
            if missing:
                raise ValueError(f"{self.config.name}: no clip {sorted(missing)} under {self.config.root}")
            pairs = [(identity, clip) for identity, clip in pairs if identity.sequence_key in self.config.sequences]
        return pairs

    def convert(self, identity: SequenceIdentity, source: BedlamClip, *, force: bool) -> Path:
        """Read the clip once (encoding it when base is pending), pose every person and move them into the camera frame, write."""
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        samples: list[SynSample] = source.samples[: self.config.frame_limit or len(source.samples)]
        with work_dir("mamma-bedlam-") as work:
            with self.timer.stage("transcode" if paths.BASE_LAYER in pending else "read"):
                (read,) = read_cameras(source.directory, {source.clip: samples}, work if paths.BASE_LAYER in pending else None, parse_bedlam)
            labels: BedlamLabels = assemble_bedlam(read.views)
            clock: Clock = Clock(labels.frames, frame_times(labels.frames, BEDLAM_NUMBERING_FPS))
            if self.body is None:
                self.body = SmplxBody(self.config.model_root)
            body: SmplxBody = self.body
            with self.timer.stage("smplx"):
                posed: list[PosedPerson] = [
                    in_camera(body.pose(person, clock.rows, mesh_stride=MESH_STRIDE), transforms) for person, transforms in zip(labels.people, labels.cam_T_body, strict=True)
                ]
                joint_error_m: float = check_joints(labels.people, posed, labels.joints3d)
            if paths.BASE_LAYER in pending:
                with (
                    self.timer.stage(f"write:{paths.BASE_LAYER}"),
                    writing.atomic_recording(targets[paths.BASE_LAYER], recording_id=identity.recording_id, default_blueprint=layout(1, ["camera"]), send_properties=True) as recording,
                ):
                    write_bedlam_base(recording, identity, source, labels, read, work, clock, joint_error_m=joint_error_m)
        assert read.size is not None
        camera: MammaCamera = MammaCamera("camera", labels.K[0], np.eye(4), *read.size, lens=None)
        self.write_layers(
            identity,
            targets,
            [layer for layer in pending if layer != paths.BASE_LAYER],
            {
                paths.BODY_POSE_LAYER: lambda recording: write_bedlam_pose(recording, labels, posed, clock),
                paths.BODY_MESH_LAYER: lambda recording: write_body_mesh(recording, posed, clock, labels.ids),
                paths.PROJECTIONS_LAYER: lambda recording: write_bedlam_projections(recording, labels, posed, clock),
                paths.MASKS_LAYER: lambda recording: write_instance_masks(recording, [camera], [read], labels.ids, BEDLAM_NUMBERING_FPS),
            },
        )
        self.timer.capture_s = (int(labels.frames[-1]) - int(labels.frames[0]) + 5) / BEDLAM_NUMBERING_FPS
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
