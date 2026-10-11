"""MAMMA layer writers: base (cameras and AV1 video), body_pose, body_mesh, projections, masks and markers.

Every stream logs on ``video_time`` = frame / fps from the sequence's own rate, beside the release ``frame_index``.
"""

import io
import tarfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters

from dataforge import hands, meshes, paths, schema, writing
from dataforge.clocks import frame_times
from dataforge.datasets.mamma_body import MODEL_FILES, PosedPerson
from dataforge.datasets.mamma_source import MARKER_FILES, MammaCamera, MammaSequence, Method, SequenceLabels
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    annotation_context,
    frame_index_column,
    layout_frustum_m,
    log_camera_node,
    log_camera_source,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, mp4_frame_count, parallel_clips, transcode_mp4, work_dir

MAX_PIXELS: int = 3_200_000
"""Videos above this many pixels are stored at half size, repeatedly (Pablo's ~3 MP cap): 4K iPhones at 1920x1080 and eval
Singles' 4112x3008 at 2056x1504; every other rig is 2056x1504 or 2048x1504 and stays native."""
SOURCE_TAGS: dict[Method, str] = {
    "mosh_v_template": "mosh_gt",
    "mosh_betas": "mosh_gt",
    "mamma_pred": "mamma_pred",
    "mamma_render": "synthetic_gt",
    "bedlam_render": "synthetic_gt",
}
"""Where a sequence's bodies come from: MoSh++ fits to Vicon markers, the MAMMA model's markerless prediction, or the
parameters a synthetic render was made from (MammaSyn)."""
MESH_BATCH_ROWS: int = 32
"""Mesh rows per chunk: one SMPL-X row is 126 KB, so a chunk is ~4 MB plus its triangle list."""
MASK_DOWNSCALE: int = 4
"""Masks are stored at a quarter of their video's width and height and scaled back over it (Pablo, 2026-10-11): an overlay
needs no more, and at the video's size they would weigh as much as the video."""
MASK_WORKERS: int = 16
"""Threads re-encoding mask PNGs. Decoding the shipped PNG is the whole cost (~11 ms at 2056x1504, ~40 ms at eval Singles'
12 MP) and OpenCV releases the GIL for it: 16 threads cut it 6.6x. Processes lose to their start-up (each imports torch)."""
PERSON_COLOURS: list[tuple[int, int, int]] = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
"""Mask colour of each person, cycled."""
MASK_OPACITY: float = 0.5
"""Opacity the viewer draws a mask with; its background palette entry is transparent. Set explicitly: at the default (1.0)
Rerun 0.38 can draw a translucent PNG's transparent pixels black."""
MARKER_COLOURS: dict[str, tuple[int, int, int]] = {"vicon": (40, 220, 40), "mosh_baseline": (255, 150, 0)}


def stored_size(width: int, height: int) -> tuple[int, int]:
    """The size a camera's video is stored at: halved until it has at most ``MAX_PIXELS``."""
    while width * height > MAX_PIXELS:
        width, height = width // 2, height // 2
    return width, height


def pinhole(camera: MammaCamera, width: int, height: int) -> PinholeParameters:
    """The camera's pinhole (K and pose) at its stored size; the Vicon lens is not part of a Rerun pinhole."""
    scale: float = width / camera.width
    world_T_cam: Float64[ndarray, "4 4"] = camera.world_T_cam
    return PinholeParameters(
        name=camera.name,
        extrinsics=Extrinsics(world_R_cam=world_T_cam[:3, :3], world_t_cam=world_T_cam[:3, 3]),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF",
            fl_x=float(camera.K[0, 0]) * scale,
            fl_y=float(camera.K[1, 1]) * scale,
            cx=float(camera.K[0, 2]) * scale,
            cy=float(camera.K[1, 2]) * scale,
            width=width,
            height=height,
        ),
    )


@dataclass(frozen=True, slots=True)
class Clock:
    """The frames a conversion writes and their times."""

    frames: Int64[ndarray, "t"]
    """Release frame indices, from 0."""
    times_ns: Int64[ndarray, "t"]
    """``frames`` / fps, nanoseconds."""

    @classmethod
    def of(cls, labels: SequenceLabels, frame_limit: int | None) -> "Clock":
        frames: Int64[ndarray, "t"] = np.arange(min(labels.frames, frame_limit or labels.frames), dtype=np.int64)
        return cls(frames, frame_times(frames, labels.fps))

    @property
    def rows(self) -> Int64[ndarray, "t"]:
        """Row of each frame in the labels' arrays (MPI: the frame itself; MammaSyn: its place among the source frames)."""
        return np.arange(len(self.frames), dtype=np.int64)

    @property
    def indexes(self) -> list[rr.TimeColumn]:
        """Both recording clocks at every frame."""
        return [time_column(self.times_ns), frame_index_column(self.frames)]


def log_cameras(recording: rr.RecordingStream, cameras: list[MammaCamera], frustum_m: float) -> tuple[list[str], list[tuple[int, int] | None]]:
    """One rig per camera (pinhole at the stored size, the Vicon lens as data where there is one); returns each camera's
    resolution summary and its rescale (None when stored native)."""
    resolutions: list[str] = []
    rescales: list[tuple[int, int] | None] = []
    for rig, camera in enumerate(cameras, start=1):
        size: tuple[int, int] = stored_size(camera.width, camera.height)
        rescale: tuple[int, int] | None = None if size == (camera.width, camera.height) else size
        rescales.append(rescale)
        log_rig_node(recording, rig, reference=None, num_cameras=1, name=camera.name, kind="exo")
        log_camera_node(
            recording,
            rig,
            0,
            pinhole(camera, *size),
            name=camera.name,
            kind="rgb",
            image_plane_distance=frustum_m,
            camera_model="pinhole" if camera.lens is None else "vicon_radial_2",
        )
        if camera.lens is not None:  # native pixels: u = c + d (1 + k1 r^2 + k2 r^4 + k3 r^6), d the distorted offset from c
            rr.log(
                schema.cam_path(rig, 0),
                rr.AnyValues(vicon_radial_2=np.concatenate([camera.lens.centre, camera.lens.radial])),
                static=True,
                recording=recording,
            )
        resolutions.append(
            log_camera_source(
                recording,
                rig,
                0,
                name=camera.name,
                source_width=camera.width,
                source_height=camera.height,
                stored_width=None if rescale is None else rescale[0],
                stored_height=None if rescale is None else rescale[1],
                video_codec="av1",
                cq=AV1_CQ,
                gop=AV1_GOP,
            )
        )
    return resolutions, rescales


def write_base(
    recording: rr.RecordingStream,
    identity: SequenceIdentity,
    sequence: MammaSequence,
    labels: SequenceLabels,
    cameras: list[MammaCamera],
    clock: Clock,
    timer: SequenceTimer,
) -> None:
    """Cameras (pinhole at the stored size, Vicon lens as data) and every camera's video, re-encoded to AV1 4:2:0."""
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
    rr.log("/", annotation_context(), static=True, recording=recording)
    resolutions, rescales = log_cameras(recording, cameras, layout_frustum_m(np.array([camera.world_T_cam[:2, 3] for camera in cameras])))
    with timer.stage("probe"):
        for camera in cameras:  # -frames:v would silently cut a longer video
            count: int = mp4_frame_count(sequence.video(camera.name))
            if count != labels.frames:
                raise ValueError(f"{sequence.video(camera.name)}: {count} frames, the sequence has {labels.frames}")
    with work_dir("mamma-") as work:
        clips: list[Path] = [work / f"rig_{rig:02d}.mp4" for rig in range(1, len(cameras) + 1)]

        def encode(camera: MammaCamera, clip: Path, rescale: tuple[int, int] | None) -> None:
            transcode_mp4(
                sequence.video(camera.name),
                clip,
                gop=AV1_GOP,
                cq=AV1_CQ,
                fps=labels.fps,
                frames=len(clock.frames),
                gray=False,
                size=rescale,
                decode="cuda",
            )

        jobs: list[tuple[Path, Callable[[], None]]] = [
            (clip, partial(encode, camera, clip, rescale)) for camera, clip, rescale in zip(cameras, clips, rescales, strict=True)
        ]
        with parallel_clips(jobs, timer, workers=6) as encoded:
            for rig, clip in enumerate(encoded, start=1):
                with timer.stage("write:video"):
                    log_video_stream(recording, clip, schema.video_path(rig, 0), times_ns=clock.times_ns, frame_indices=clock.frames)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(clock.frames),
        num_cameras=len(cameras),
        source_num_frames=labels.frames,
        fps=labels.fps,
        clock_source=f"release frame index / {labels.fps} fps (global.npz fps); the release has no capture timestamps",
        source_resolution=pa.array(resolutions),
        video_source=f"AV1 NVENC re-encode (CQ {AV1_CQ}, 4:2:0) of the release's HEVC 4:4:4 {sequence.video_dir.name}; videos above "
        f"{MAX_PIXELS / 1e6:.1f} MP stored at half size (the originals stay in the MAMMA release)",
    )
    recording.send_property(
        "episode",
        rr.AnyValues(
            collection=sequence.collection,
            sequence=labels.name,
            body_source=SOURCE_TAGS[labels.method],
            fit_method=labels.method,
            num_people=len(labels.people),
            subjects=pa.array([person.subject for person in labels.people]),
        ),
    )


def write_body_pose(recording: rr.RecordingStream, labels: SequenceLabels, posed: list[PosedPerson], clock: Clock) -> None:
    """Every person's SMPL-X parameters at full rate, as shipped, and the COCO-133 keypoints of the posed model; NaN rows where
    a person is absent."""
    indexes: list[rr.TimeColumn] = clock.indexes
    for index, (person, pose) in enumerate(zip(labels.people, posed, strict=True)):
        rr.log(
            schema.person_path(index),
            rr.AnyValues(
                subject=person.subject,
                gender=person.gender,
                source=SOURCE_TAGS[labels.method],
                fit_method=labels.method,
                model=MODEL_FILES[person.model].path,
            ),
            static=True,
            recording=recording,
        )
        path: str = schema.person_body_path(index, "smplx")
        rr.log(
            path,
            rr.AnyValues(
                drop_untyped_nones=True,
                betas=person.betas,
                v_template=None if person.v_template is None else person.v_template.reshape(-1),
                flat_hand_mean=person.flat_hand_mean,
                pose_layout="165 axis-angle: root, 21 body, jaw, left eye, right eye, 15 left hand, 15 right hand",
                translation_pivot="root joint (smplx transl)",
            ),
            static=True,
            recording=recording,
        )
        present: Bool[ndarray, "t 1"] = (np.ones(len(clock.rows), dtype=bool) if person.present is None else person.present[clock.rows])[:, None]
        pose_rows: Float64[ndarray, "t 165"] = np.where(present, person.pose[clock.rows], np.nan)
        trans_rows: Float64[ndarray, "t 3"] = np.where(present, person.trans[clock.rows], np.nan)
        columns = rr.AnyValues.columns(
            pose=pa.FixedSizeListArray.from_arrays(pa.array(pose_rows.reshape(-1)), 165),
            trans=pa.FixedSizeListArray.from_arrays(pa.array(trans_rows.reshape(-1)), 3),
        )
        rr.send_columns(path, indexes=indexes, columns=columns, recording=recording)
        hands.log_keypoints3d(
            recording,
            times_ns=clock.times_ns,
            frame_indices=clock.frames,
            positions=pose.coco133,
            confidence=None,
            path=schema.person_coco133_xyz_path(index),
        )


def write_body_mesh(recording: rr.RecordingStream, posed: list[PosedPerson], clock: Clock) -> None:
    """Every person's posed SMPL-X mesh at its mesh rows, an empty mesh where the person is absent."""
    for index, pose in enumerate(posed):
        path: str = schema.person_body_path(index, "mesh")
        meshes.log_mesh_static(recording, path, albedo_factor=hands.BODY_ALBEDO)
        times: Int64[ndarray, "k"] = clock.times_ns[pose.mesh_rows]
        frames: Int64[ndarray, "k"] = clock.frames[pose.mesh_rows]
        before: Int64[ndarray, "k"] = np.cumsum(pose.mesh_present) - pose.mesh_present  # posed meshes ahead of each row
        for start in range(0, len(pose.mesh_rows), MESH_BATCH_ROWS):
            rows: slice = slice(start, start + MESH_BATCH_ROWS)
            trusted: list[bool] = pose.mesh_present[rows].tolist()
            first: int = int(before[start])
            meshes.log_mesh_batch(
                recording,
                path,
                times_ns=times[rows],
                frame_indices=frames[rows],
                vertices=pose.vertices[first : first + sum(trusted)],
                trusted=trusted,
                topology=pose.faces,
            )


def write_projections(
    recording: rr.RecordingStream, cameras: list[MammaCamera], posed: list[PosedPerson], clock: Clock, *, calibration_source: str
) -> None:
    """Every person's COCO-133 keypoints through every camera's lens (Vicon radial, or pinhole), in stored pixels."""
    models: list[str] = sorted({"pinhole" if camera.lens is None else "vicon_radial_2" for camera in cameras})
    recording.send_property(
        paths.PROJECTIONS_LAYER, rr.AnyValues(derived_from="coco133_xyz", camera_model=", ".join(models), calibration_source=calibration_source)
    )
    for rig, camera in enumerate(cameras, start=1):
        scale: float = stored_size(camera.width, camera.height)[0] / camera.width
        for index, pose in enumerate(posed):
            pixels: Float64[ndarray, "n 2"] = camera.project(pose.coco133.reshape(-1, 3).astype(np.float64)) * scale
            positions: Float32[ndarray, "t 133 2"] = pixels.reshape(-1, 133, 2).astype(np.float32)
            hands.log_keypoints2d(
                recording,
                schema.person_projection_path(rig, 0, index),
                times_ns=clock.times_ns,
                frame_indices=clock.frames,
                positions=positions,
                confidence=None,  # 1 where the lens put the point, 0 where it is NaN
            )


def palette_png(gray: ndarray, size: tuple[int, int], colour: tuple[int, int, int]) -> bytes:
    """A 0/255 mask as a two-colour palette PNG at ``size``: index 1 inside (``colour``), index 0 transparent outside.

    A pixel is inside when most of the source pixels it covers are. The palette indices are the mask; a viewer draws the colours.
    """
    # No jaxtyping annotations on these locals: this runs on MASK_WORKERS threads, and the dev env's per-call check of a fresh
    # jaxtyping hint grows beartype's cache from every thread at once ("dictionary changed size during iteration").
    if gray.shape[::-1] != size:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    ids = (gray >= 128).astype(np.uint8)
    paletted: Image.Image = Image.frombuffer("P", size, ids, "raw", "P", 0, 1)
    paletted.putpalette([0, 0, 0, *colour])
    buffer = io.BytesIO()
    paletted.save(buffer, format="PNG", transparency=bytes([0, 255]))
    return buffer.getvalue()


def decode_gray(source: bytes) -> ndarray:
    """An 8-bit gray PNG, decoded (OpenCV releases the GIL for it)."""
    gray = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("not a decodable image")
    return gray


def mask_png(source: bytes, size: tuple[int, int], colour: tuple[int, int, int]) -> bytes:
    """A shipped 0/255 mask of one person as a palette PNG at ``size`` (``palette_png``)."""
    return palette_png(decode_gray(source), size, colour)


def log_person_masks(
    recording: rr.RecordingStream, rig: int, person: int, stored: tuple[int, int], size: tuple[int, int], blobs: list[bytes], indexes: list[rr.TimeColumn]
) -> None:
    """One person's mask PNGs in one camera, at ``size``, scaled over the camera's stored video."""
    path: str = schema.person_mask_path(rig, 0, person)
    rr.log(
        path,
        rr.EncodedImage.from_fields(media_type="image/png", opacity=MASK_OPACITY, draw_order=10.0 + person),
        static=True,
        recording=recording,
    )
    # The scale lays the small mask over the stored video in the pinhole's image space.
    rr.log(path, rr.Transform3D(scale=[stored[0] / size[0], stored[1] / size[1], 1.0]), static=True, recording=recording)
    rr.send_columns(path, indexes=indexes, columns=rr.EncodedImage.columns(blob=blobs), recording=recording)


def write_masks(recording: rr.RecordingStream, sequence: MammaSequence, cameras: list[MammaCamera], people: int, clock: Clock) -> None:
    """Every camera's shipped per-person masks (``mask_<frame>_<person>.png``, person from 1), one palette PNG per person and frame."""
    recording.send_property(
        paths.MASKS_LAYER,
        rr.AnyValues(
            encoding=f"palette PNG at 1/{MASK_DOWNSCALE} of the stored video's width and height, scaled over it by a Transform3D: "
            "index 1 = the person, index 0 = background (transparent)",
            person_ids="person_NN = the release's mask _NN+1",
        ),
    )
    indexes: list[rr.TimeColumn] = clock.indexes
    with ThreadPoolExecutor(max_workers=MASK_WORKERS) as executor:
        for rig, camera in enumerate(cameras, start=1):
            stored: tuple[int, int] = stored_size(camera.width, camera.height)
            size: tuple[int, int] = (stored[0] // MASK_DOWNSCALE, stored[1] // MASK_DOWNSCALE)
            tar_path: Path = sequence.mask_tar(camera.name)
            with tarfile.open(tar_path) as tar:
                members: dict[str, tarfile.TarInfo] = {member.name: member for member in tar.getmembers()}
                for person in range(people):
                    names: list[str] = [f"mask_{frame:04d}_{person + 1:02d}.png" for frame in clock.frames]
                    missing: list[str] = [name for name in names if name not in members]
                    if missing:
                        raise ValueError(f"{tar_path}: no {missing[:3]} ({len(missing)} of {len(names)} masks of person {person + 1} missing)")
                    sources: list[bytes] = [tar.extractfile(members[name]).read() for name in names]  # type: ignore[union-attr]
                    colour: tuple[int, int, int] = PERSON_COLOURS[person % len(PERSON_COLOURS)]
                    blobs: list[bytes] = list(executor.map(partial(mask_png, size=size, colour=colour), sources))
                    log_person_masks(recording, rig, person, stored, size, blobs, indexes)


def write_markers(recording: rr.RecordingStream, sequence: MammaSequence, clock: Clock) -> None:
    """The held-out Vicon markers and the MoSh++ baseline's markers (eval Extra), labelled, world metres."""
    names, vicon, baseline = MARKER_FILES
    labels: list[str] = [str(label) for label in np.load(sequence.markers_dir / names, allow_pickle=False)]
    indexes: list[rr.TimeColumn] = clock.indexes
    for kind, name in (("vicon", vicon), ("mosh_baseline", baseline)):
        positions: Float32[ndarray, "f m 3"] = np.load(sequence.markers_dir / name, allow_pickle=False)
        if positions.shape[1:] != (len(labels), 3) or len(positions) < len(clock.frames):
            raise ValueError(f"{sequence.markers_dir / name}: shape {positions.shape} for {len(labels)} labels and {len(clock.frames)} frames")
        path: str = schema.markers_path(kind)
        rr.log(
            path,
            rr.Points3D.from_fields(labels=labels, colors=MARKER_COLOURS[kind], radii=0.008, show_labels=False),
            static=True,
            recording=recording,
        )
        rr.send_columns(
            path,
            indexes=indexes,
            columns=rr.Points3D.columns(positions=positions[clock.frames].reshape(-1, 3)).partition([len(labels)] * len(clock.frames)),
            recording=recording,
        )
