"""MammaSyn, MAMMA's synthetic training set, rebuilt from its WebDataset shards into the MPI families' layers.

``--family syn`` converts MAMMA's own renders: one recording per sequence directory, every camera's PNG frames encoded to an
AV1 video, the parameters the render was made from as body_pose, and its instance masks. ``--family syn-bedlam`` (the BEDLAM
release repacked) is the other family. The shards are read in place and never modified. See ``docs/mamma.md``.
"""

import json
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import serde

from dataforge import blueprints, paths, schema, writing
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
from dataforge.datasets.mamma_source import SequenceLabels, decode
from dataforge.datasets.mamma_syn_source import (
    RENDER_FPS,
    RenderLabels,
    RenderView,
    ShardReader,
    SynSample,
    SynSequence,
    assemble_render,
    discover_renders,
    read_render_view,
    sequence_samples,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import annotation_context, frame_index_column, layout_frustum_m, log_video_stream, time_column
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, encode_frames_to_mp4, work_dir

JOINT_TOLERANCE_M: float = 1e-3
"""Largest gap allowed between the shipped joints3d and the same joints posed from the shipped parameters. Measured on the
release: 0.00 mm (the renders were made from exactly these parameters with the locked-head neutral)."""
MESH_STRIDE: int = 1
"""Every frame keeps its mesh: at 6 fps a coarser stride would leave the body still for half a second."""


@serde
@dataclass(frozen=True, slots=True)
class SynMetadata:
    """A sequence directory's ``metadata.json`` (third party: unknown fields allowed)."""

    format: str
    num_samples: int
    """Camera-view samples in the directory's shards."""


@dataclass
class CameraRead:
    """One camera's pass over its samples: its labels, its masks and its image size."""

    camera: str
    frames: list[int] = field(default_factory=list)
    views: list[RenderView] = field(default_factory=list)
    masks: list[bytes] = field(default_factory=list)
    size: tuple[int, int] | None = None


def png_size(payload: bytes, origin: str) -> tuple[int, int]:
    """Width and height from a PNG's header."""
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{origin}: not a PNG")
    return int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")


def stream_camera(directory: Path, samples: list[SynSample], read: CameraRead) -> Iterator[bytes]:
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
            read.views.append(RenderView(sample.camera, sample.frame, read_render_view(shards.read(sample.labels), origin)))
            yield image


def read_cameras(directory: Path, by_camera: dict[str, list[SynSample]], work: Path | None) -> list[CameraRead]:
    """Every camera's pass, in parallel; with ``work``, each camera's renders are encoded to ``work/<camera>.mp4`` on the way."""

    def run(camera: str) -> CameraRead:
        read: CameraRead = CameraRead(camera)
        frames: Iterator[bytes] = stream_camera(directory, by_camera[camera], read)
        if work is None:
            for _ in frames:
                pass
        else:
            encoded: int = encode_frames_to_mp4(frames, work / f"{camera}.mp4", source=FrameSource("png"), fps=RENDER_FPS, gop=AV1_GOP, cq=AV1_CQ)
            if encoded != len(by_camera[camera]):
                raise ValueError(f"{directory}/{camera}: {encoded} frames encoded of {len(by_camera[camera])}")
        return read

    with ThreadPoolExecutor(max_workers=len(by_camera)) as executor:
        return list(executor.map(run, sorted(by_camera)))


def check_joints(people_posed: list[PosedPerson], labels: RenderLabels) -> float:
    """The largest gap between each person's shipped joints3d and the posed ones; raises above ``JOINT_TOLERANCE_M``."""
    worst: float = 0.0
    for person, posed, shipped in zip(labels.people, people_posed, labels.joints3d, strict=True):
        gap: float = float(np.nanmax(np.abs(posed.joints[:, :127] - shipped))) if np.isfinite(shipped).any() else 0.0
        if gap > JOINT_TOLERANCE_M:
            raise ValueError(f"{person.subject}: the shipped parameters miss the shipped joints3d by {gap * 1000:.1f} mm")
        worst = max(worst, gap)
    return worst


def camera_clock(read: CameraRead) -> Clock:
    frames: Int64[ndarray, "n"] = np.array(read.frames, dtype=np.int64)
    return Clock(frames, frame_times(frames, RENDER_FPS))


def write_render_base(
    recording: rr.RecordingStream,
    identity: SequenceIdentity,
    sequence: SynSequence,
    labels: RenderLabels,
    reads: list[CameraRead],
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
        clock: Clock = camera_clock(read)
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


def instance_mask_pngs(source: bytes, size: tuple[int, int], keys: list[int]) -> list[bytes]:
    """One palette PNG per person (``keys``, in order) from a sample's instance mask, where pixel value = person index + 1."""
    ids = decode_gray(source)  # decoded once for every person; no jaxtyping on thread-pool locals (see palette_png)
    return [
        palette_png(np.where(ids == key + 1, 255, 0).astype(np.uint8), size, PERSON_COLOURS[index % len(PERSON_COLOURS)])
        for index, key in enumerate(keys)
    ]


def write_render_masks(recording: rr.RecordingStream, labels: RenderLabels, reads: list[CameraRead]) -> None:
    """Every camera's instance masks split per person (mask value = person index + 1), as palette PNGs at a quarter size."""
    recording.send_property(
        paths.MASKS_LAYER,
        rr.AnyValues(
            encoding=f"palette PNG at 1/{MASK_DOWNSCALE} of the stored video's width and height, scaled over it by a Transform3D: "
            "index 1 = the person, index 0 = background (transparent)",
            person_ids="person_NN = the shards' instance mask value NN + 1",
        ),
    )
    keys: list[int] = [int(person.subject.removeprefix("person_")) for person in labels.people]
    with ThreadPoolExecutor(max_workers=MASK_WORKERS) as executor:
        for rig, (camera, read) in enumerate(zip(labels.cameras, reads, strict=True), start=1):
            clock: Clock = camera_clock(read)
            stored: tuple[int, int] = stored_size(camera.width, camera.height)
            size: tuple[int, int] = (stored[0] // MASK_DOWNSCALE, stored[1] // MASK_DOWNSCALE)
            frames: list[list[bytes]] = list(executor.map(partial(instance_mask_pngs, size=size, keys=keys), read.masks))
            for index in range(len(keys)):
                blobs: list[bytes] = [people[index] for people in frames]
                log_person_masks(recording, rig, index, stored, size, blobs, [time_column(clock.times_ns), frame_index_column(clock.frames)])


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
            shipped: SynMetadata = decode(SynMetadata, source.metadata, json.loads(source.metadata.read_text()))
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
                reads: list[CameraRead] = read_cameras(source.directory, by_camera, work if paths.BASE_LAYER in pending else None)
            labels: RenderLabels = assemble_render([view for read in reads for view in read.views], {read.camera: read.size for read in reads if read.size})
            clock: Clock = Clock(labels.frames, frame_times(labels.frames, RENDER_FPS))
            if self.body is None:
                self.body = SmplxBody(self.config.model_root)
            body: SmplxBody = self.body
            with self.timer.stage("smplx"):
                posed: list[PosedPerson] = [body.pose(person, clock.rows, mesh_stride=MESH_STRIDE) for person in labels.people]
                joint_error_m: float = check_joints(posed, labels)
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
        self.write_layers(
            identity,
            targets,
            [layer for layer in pending if layer != paths.BASE_LAYER],
            {
                paths.BODY_POSE_LAYER: lambda recording: write_body_pose(recording, sequence_labels, posed, clock),
                paths.BODY_MESH_LAYER: lambda recording: write_body_mesh(recording, posed, clock),
                paths.PROJECTIONS_LAYER: lambda recording: write_projections(
                    recording, labels.cameras, posed, clock, calibration_source="each sample's label pickle (cam_int, cam_ext)"
                ),
                paths.MASKS_LAYER: lambda recording: write_render_masks(recording, labels, reads),
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
