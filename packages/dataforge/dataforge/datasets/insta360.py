"""Insta360 dual-fisheye ``.insv`` clips: one base recording per clip, and on request a derived panorama layer.

The base is ``insta360_io.InsvReader``'s chunk stream (packages/insta360-io; the reader's module docstring states its
entities, the device profile and the clock) with ``entity_path`` the rig ``/world/rig_00`` and ``timeline_name``
``video_time`` (``capture:start_time_ns`` is its zero), apart from the lens videos and cameras: each lens is stored
re-encoded to AV1 at most 1920 pixels wide (one halving on the X5, an exception to dataforge's ~3 MP cap) and its Mei
camera scaled to that size (``insta360_lens_video``). ``cam_00`` and ``cam_01`` are lens 0 and lens 1, whichever track
holds them (``stream_id``). This module adds the exoego:v2 tags (rig node, sensor names, camera sources), the IMU noise
model and the capture properties. A clip without a device profile gets no rig reference and no
IMU noise model, so slam-rs refuses it. The IMU noise model is a placeholder, marked as such in its ``source``.

Panorama layer (``--panorama``; off by default). ``panorama`` holds the RGB equirectangular panorama of every
frameset at ``/world/rig_00/pano`` (``insta360_panorama``: ``W x W/2``, by default ``W`` = twice the lens width, 3840x1920
at 1920), encoded AV1 (NVENC, else SVT-AV1), on the frame times. Each pixel reads its own lens; in a seam band
(``pano_band_deg``, default 5 deg either side of the seam plane) the two lenses are blended, on ``pano_device`` ``cuda``
(the default when torch sees a GPU; the dataforge-cuda env) across an optical-flow seam (NVIDIA's flow engine) so near
objects do not ghost, and on ``cpu`` with the plain feather. On ``cuda`` the stage stays on the GPU: NVDEC decode,
remap, flow, blend, NVENC from device memory. It is a derived
output: nothing reads it as input, slam-rs tracks the two lens cameras of the base, and the lens brightness is not
matched beyond the feather. The tables come from the base recording's two cameras as written (float32 statics), so a
later ``--panorama`` run adds the layer beside an existing base without writing the base again.

Cube layer (``--cube``; off by default). ``cube`` holds the six-face cubemap of ``insta360_cube`` for every
``cube_stride``-th frameset: ``N x N`` pinholes at ONE centre (the midpoint of the two lens centres), cube axes = camera
0's, side faces joined at their centre line from the two lens images (no blend). ``/world/rig_00/cam_02 .. cam_07`` =
``cube6_front0``, ``cube6_front1``, ``cube6_left``, ``cube6_right``, ``cube6_up``, ``cube6_down``, each with its name on
the node, a static face-from-IMU ``Transform3D`` (``ChildFromParent``), a plain ``Pinhole`` (RDF) and an RGB AV1
``VideoStream`` on the source frameset's stamp. The tables come from the base recording's two cameras through
``MeiCamera.project`` and the pictures from its lens videos, so the layer can be added beside an existing base. On ``cube_device`` ``cuda`` (the default when torch sees a GPU) the
stage stays on the GPU (the dataforge-cuda env): NVDEC decode, one remap per face, NVENC from device memory (SVT-AV1 from
host memory where NVENC does not encode AV1, e.g. Ampere); ``cpu`` decodes with PyAV and remaps with OpenCV behind the
same API, and encodes on NVENC when the host has it, else SVT-AV1.

Panorama frame ``E``. Axes = camera 0's (RDF), origin = the midpoint of the two camera centres. Pixel ``(u, v)`` of a
``W``-wide panorama (pixel centres at integers) looks along longitude ``lon = (u + 0.5) 2 pi / W - pi / 2``,
``lon = atan2(x, z)`` in ``E``, and latitude ``lat = (v + 0.5) 2 pi / W - pi / 2``, ``+y`` down: the left half is camera
0's hemisphere (centre column ``W/4`` = camera 0's ``+Z``), the right half camera 1's (column ``3W/4`` = ``-Z``). Both
lenses are read from the rig centre (only camera 1's rotation in camera 0 enters the tables), so outside the seam band
the panorama ignores the lens-centre baseline (33.3 mm on the X4 Air, 31.9 mm on the X5); in the band the flow seam
takes it up.

From the native poses to the panorama. ``E`` is fixed in the rig: ``T_cam0_E = (I, c1 / 2)``, with ``c1`` camera 1's
centre in camera 0 (the translation of ``cam0_T_cam1``, from the ``cam_00`` and ``cam_01`` ``Transform3D``s,
``ChildFromParent``, as ``cameras_from_recording`` reads them). A COLMAP ``EQUIRECTANGULAR`` camera ``C`` (single
centre; ``u = W (lon_C + pi) / (2 pi)`` in pixel-corner coordinates, image centre = ``+Z_C``, ``+y`` down) sees this
panorama with its columns a quarter turn on, ``lon_E = lon_C + pi / 2``, so ``T_cam0_C = (Ry(+90 deg), c1 / 2)`` with
``Ry(+90 deg) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]`` (``+Z_C`` is camera 0's ``+X``). slam-rs reports ``world_from_rig``
in the IMU frame ``imu_00`` (the rig reference), so a tracked pose ``T_world_imu`` becomes
``T_world_C = T_world_imu * T_imu_cam0 * T_cam0_C`` (COLMAP's ``cam_from_world`` is its inverse), with ``T_imu_cam0``
the inverse of the ``cam_00`` ``Transform3D``. Every factor is a static of the base, so the transform needs no file of
its own; with the profiles' zero lever arm the midpoint, and so the panorama centre, is the IMU origin.
"""

from __future__ import annotations

import contextlib
import functools
import io
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import insta360_io
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
import torch
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.camera_parameters import MeiParameters
from simplecv.catalog_calibration import CatalogCameraCalibration, read_camera_models_from_store
from simplecv.imu_calibration import ImuCalibration
from simplecv.ops.gpu_image import area_downscale_uint8
from torch import Tensor

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets import insta360_cube as cube
from dataforge.datasets import insta360_panorama
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.datasets.insta360_blueprint import FACE_CAMS, FIRST_FACE_CAM, PANO_ENTITY, RIG, segment_layout, segment_walk
from dataforge.datasets.insta360_lens_video import LENS_FPS, LENS_MAX_SIZE, BaseLensVideos, lens_pictures, read_base_lens_videos
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import log_camera_source, log_rig_node, log_video_stream
from dataforge.video_encoding import AV1_CQ, AV1_GOP, NVENC_PRESET, nvenc_av1_options, nvenc_slot, work_dir

PANORAMA_LAYER: str = "panorama"
"""The derived panorama, written only under ``--panorama``: an AV1 ``VideoStream`` at ``PANO_ENTITY`` on the frame times."""
CUBE_LAYER: str = "cube"
"""The cubemap, written only under ``--cube``: six faces per sampled frameset, ``cam_02 .. cam_07``."""
LENS_BATCH_PIXELS: int = 16 * 1920 * 1920
"""Lens pixels the cube and panorama layers decode, join and encode per step: 16 framesets at 1920, 4 at 3840."""
RUN_SOURCE: str = schema.SLAM_RS_SOURCE
"""The run layer the segment-table card shows: slam-rs tracks these clips."""


@dataclass(frozen=True, slots=True)
class LensCameras:
    """The two lens cameras of a base recording as written (float32 statics)."""

    cameras: list[insta360_io.MeiCamera]
    """Lens 0 and lens 1."""
    lens_from_rig: list[Float64[ndarray, "4 4"]]
    """``T_lens_imu`` of each lens, as the ``ChildFromParent`` ``Transform3D`` states it."""
    size: tuple[int, int]
    """Lens image width and height."""

    @property
    def lens1_from_lens0(self) -> Float64[ndarray, "3 3"]:
        """Lens 1's rotation from lens 0."""
        return self.lens_from_rig[1][:3, :3] @ self.lens_from_rig[0][:3, :3].T


def cameras_from_recording(base: Path) -> LensCameras:
    """A base recording's ``cam_00`` and ``cam_01``, read through simplecv's calibration decoder: each its Mei model
    (either name, ``simplecv.camera_parameters.MEI_CATALOG_MODELS``) and pose.

    Raises:
        ValueError: If a camera is not a Mei camera or its statics are missing.
    """
    nodes: list[str] = [schema.cam_path(RIG, cam) for cam in range(2)]
    models: dict[str, CatalogCameraCalibration] = read_camera_models_from_store(rrc.RrdReader(base).store(), nodes, str(base))
    cameras: list[insta360_io.MeiCamera] = []
    for node in nodes:
        # The typed model reads either Mei name: gamma from the Pinhole (mei_v6) or from the coefficients (mei_v6_unified,
        # whose Pinhole is the viewer's paraxial focal).
        mei: MeiParameters | None = models[node].mei
        if mei is None:
            raise ValueError(f"{base}: {node} has no Mei distortion model, so it is not a Mei camera")
        cameras.append(insta360_io.MeiCamera(mei.model, mei.gamma_x, mei.gamma_y, mei.cx, mei.cy, mei.sphere_coefficients()))
    lens_from_rig: list[Float64[ndarray, "4 4"]] = [np.asarray(models[node].pinhole.extrinsics.cam_T_world, dtype=np.float64) for node in nodes]
    intrinsics = models[nodes[0]].pinhole.intrinsics
    return LensCameras(cameras=cameras, lens_from_rig=lens_from_rig, size=(int(intrinsics.width), int(intrinsics.height)))


def layer_device(choice: Literal["cuda", "cpu"] | None) -> torch.device:
    """A layer's device: the configured one, else ``cuda`` when torch sees a GPU."""
    return torch.device(choice or ("cuda" if torch.cuda.is_available() else "cpu"))


def cube_indices(frames: int, stride: int, limit: int | None) -> Int64[ndarray, " k"]:
    """Framesets the cube layer samples: ``0, stride, 2 stride, ...``, the first ``limit`` of them.

    Raises:
        ValueError: If ``stride`` or ``limit`` is not positive.
    """
    if stride < 1 or (limit is not None and limit < 1):
        raise ValueError(f"cube stride {stride} and limit {limit} must be positive")
    return np.arange(0, frames, stride, dtype=np.int64)[:limit]


SVTAV1_PRESET: int = 10
"""SVT-AV1 preset of the software face encode (0 slowest .. 13 fastest)."""


@functools.cache
def av1_nvenc_encodes() -> bool:
    """Whether NVENC encodes AV1 on this host: one frame through torchcodec, in an encoder slot (a listed encoder is not
    enough: a missing driver or an older GPU fails only when the session opens)."""
    from torchcodec.encoders import Encoder

    encoder = Encoder()
    stream = encoder.add_video(height=256, width=256, frame_rate=1, device="cpu", codec="av1_nvenc")
    try:
        with nvenc_slot(), encoder.open_file_like(io.BytesIO(), format="mp4"):
            stream.add_frames(torch.zeros((1, 3, 256, 256), dtype=torch.uint8))
    except RuntimeError:
        return False
    return True


def av1_codec() -> Literal["av1_nvenc", "libsvtav1"]:
    """The AV1 encoder of a layer: NVENC where the host's NVENC encodes AV1, else SVT-AV1. A CUDA GPU is not enough:
    NVENC encodes AV1 from Ada (RTX 40xx) on, so an Ampere card such as the RTX 3060 joins on ``cuda`` and encodes in
    software (``encode_videos`` then feeds the encoder from host memory)."""
    return "av1_nvenc" if av1_nvenc_encodes() else "libsvtav1"


def encode_videos(
    batches: Iterable[UInt8[Tensor, "f b 3 h w"]],
    videos: list[Path],
    *,
    width: int,
    height: int,
    gop: int,
    cq: int,
    device: torch.device,
    codec: Literal["av1_nvenc", "libsvtav1"],
) -> int:
    """Encode batches of ``width`` x ``height`` RGB pictures into one AV1 mp4 per stream (a cube face, the panorama;
    yuv420p, a keyframe every ``gop``), fed from ``device``.

    ``av1_nvenc``: one NVENC session per stream (preset p4, vbr at ``cq``), every encoder slot reserved before the first
    encoder opens, so a busy host waits for all of them at once instead of opening some and stalling the rest.
    ``libsvtav1``: the software encoder at CRF ``cq``, for a host whose NVENC does not encode AV1; pictures on a CUDA
    ``device`` are copied to host memory for it.

    Returns:
        Frames each encoder took.

    Raises:
        ValueError: If ``DATAFORGE_NVENC_SLOTS`` is below the number of streams.
    """
    from torchcodec.encoders import Encoder

    settings: dict[str, object] = (
        {"preset": NVENC_PRESET, "extra_options": nvenc_av1_options(gop=gop, cq=cq)} if codec == "av1_nvenc" else {"preset": SVTAV1_PRESET, "crf": cq, "extra_options": {"g": gop}}
    )
    feed: torch.device = device if codec == "av1_nvenc" else torch.device("cpu")
    fed: int = 0
    with nvenc_slot(len(videos)) if codec == "av1_nvenc" else contextlib.nullcontext(), contextlib.ExitStack() as opened:
        video_streams = []
        for video in videos:
            encoder = Encoder()
            video_streams.append(encoder.add_video(height=height, width=width, frame_rate=LENS_FPS, device=feed, codec=codec, **settings))  # pyrefly: ignore[bad-argument-type]
            opened.enter_context(encoder.open_file(video))
        for batch in batches:
            for video_stream, pictures in zip(video_streams, batch, strict=True):
                video_stream.add_frames(pictures.to(feed))
            fed += batch.shape[1]
    return fed


def encode_and_log(
    recording: rr.RecordingStream,
    batches: Iterable[UInt8[Tensor, "f b 3 h w"]],
    entities: list[str],
    times_ns: Int64[ndarray, " n"],
    *,
    width: int,
    height: int,
    gop: int,
    cq: int,
    device: torch.device,
) -> None:
    """Encode the streams of ``batches`` (``encode_videos``, the host's AV1 encoder) and log stream ``k`` at
    ``entities[k]`` as a ``VideoStream`` on ``times_ns``, one picture per stamp.

    Raises:
        ValueError: If an encoder takes or a stream holds another number of pictures than ``times_ns`` has stamps.
    """
    with work_dir("insta360-encode-") as work:
        videos: list[Path] = [work / f"stream{k}.mp4" for k in range(len(entities))]
        fed: int = encode_videos(batches, videos, width=width, height=height, gop=gop, cq=cq, device=device, codec=av1_codec())
        if fed != times_ns.size:
            raise ValueError(f"the encoders took {fed} pictures for {times_ns.size} stamps ({entities[0]} ...)")
        for entity, video in zip(entities, videos, strict=True):
            samples: int = log_video_stream(recording, video, entity, times_ns=times_ns)
            if samples != times_ns.size:
                raise ValueError(f"{entity} holds {samples} pictures for {times_ns.size} stamps")


@dataclass(frozen=True, slots=True)
class ClipSource:
    """One ``.insv`` clip on disk."""

    path: Path
    """The clip."""


@dataclass
class Insta360Config(FrameLimitedConfig):
    """Insta360 dual-fisheye ``.insv`` clips (local device captures) as a two-camera + IMU rig."""

    command: ClassVar[str] = "insta360"
    """Registry key and CLI subcommand."""

    _target: type = field(default_factory=lambda: Insta360Dataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "insta360")
    """Directory holding the ``.insv`` clips; each clip is one sequence."""
    panorama: bool = False
    """Also write the panorama layer (RGB equirectangular, flow seam, AV1 NVENC). Run again with it to add the layer
    beside bases already written."""
    pano_width: int | None = None
    """Panorama width ``W`` (a multiple of 32; height ``W/2``); None = twice the lens width, its own density (3840 at 1920)."""
    pano_band_deg: float = insta360_panorama.SEAM_BAND_DEG
    """Half-width of the seam band, degrees either side of the seam plane, where the two lenses are blended."""
    pano_device: Literal["cuda", "cpu"] | None = None
    """Where the panorama is decoded and joined: ``cuda`` (NVDEC, CUDA remap, flow seam on NVOF, NVENC from device
    memory; the dataforge-cuda env) or ``cpu`` (PyAV, ``cv2.remap``; no NVOF there, so the band is the plain feather;
    NVENC when the host has it, else SVT-AV1). None = ``cuda`` when torch sees a GPU, else ``cpu``."""
    decode_threads: int = 4
    """PyAV threads per lens decoder of the panorama and cube layers on ``cpu``."""
    pano_cq: int = 34
    """NVENC constant-quality target of the panorama: a viewing layer, not an analysis input."""
    pano_gop: int = 30
    """Panorama keyframe interval in frames, about one second."""
    cube: bool = False
    """Also write the cube layer: the six-face cubemap per sampled frameset (RGB remap + AV1 NVENC)."""
    cube_stride: int = 10
    """The cube layer samples every this-many framesets, from frameset 0."""
    cube_max_framesets: int | None = None
    """Keep only the first this-many sampled framesets (a sample pack); None keeps all."""
    cube_face_size: int | None = None
    """Face side ``N`` (a multiple of 32); None = ``lens_size * 800 / 1920``."""
    cube_cq: int = 14
    """NVENC constant-quality target of the faces: they feed SfM and splat training, so near visually lossless."""
    cube_gop: int = 10
    """Face keyframe interval in sampled framesets."""
    lens_device: Literal["cuda", "cpu"] | None = None
    """Where the base's lens videos are decoded, halved and encoded: ``cuda`` (NVDEC, CUDA box filter, NVENC from device
    memory, or SVT-AV1 where NVENC does not encode AV1; the dataforge-cuda env) or ``cpu`` (PyAV, torch on the host,
    NVENC when the host has it, else SVT-AV1). None: ``cuda`` when torch sees a GPU."""
    cube_device: Literal["cuda", "cpu"] | None = None
    """Where the faces are decoded and joined: ``cuda`` (NVDEC, CUDA remap, NVENC from device memory, or SVT-AV1 where NVENC
    does not encode AV1; the dataforge-cuda env) or ``cpu`` (PyAV, OpenCV remap; NVENC when the host has it, else software AV1). None: ``cuda`` when torch sees a GPU."""


class Insta360Dataset(DataforgeDataset[Insta360Config, ClipSource]):
    """Converts ``.insv`` clips into a base layer (lens videos, IMU, calibration) and, on request, a panorama layer."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, PANORAMA_LAYER, CUBE_LAYER)

    def download(self) -> None:
        """Local device captures: check that the clip directory holds a clip."""
        if not any(self.config.root.glob("*.insv")):
            raise FileNotFoundError(f"no .insv clip in {self.config.root}: copy the camera's clips there")

    def discover(self) -> list[tuple[SequenceIdentity, ClipSource]]:
        """Every ``.insv`` clip directly in the root, named by its file stem; a two-file clip (``…_00_…`` holds lens 0
        and the trailer, ``…_10_…`` lens 1, as ``insta360_io`` pairs them) is one clip, named by its ``_00_`` file."""
        paths: list[Path] = sorted(self.config.root.glob("*.insv"))
        names: set[str] = {path.name for path in paths}

        def is_lens_1_member(path: Path) -> bool:
            parts: list[str] = path.name.split("_")
            field: int | None = next((i for i, part in enumerate(parts) if part in ("00", "10")), None)
            return field is not None and parts[field] == "10" and "_".join([*parts[:field], "00", *parts[field + 1 :]]) in names

        return [(SequenceIdentity(self.config.name, (path.stem,)), ClipSource(path)) for path in paths if not is_lens_1_member(path)]

    def convert(self, identity: SequenceIdentity, source: ClipSource, *, force: bool) -> Path:
        """Write the pending layers of one clip: the base, and the panorama under ``--panorama``."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        if not self.config.panorama:
            pending = [layer for layer in pending if layer != PANORAMA_LAYER]
        if not self.config.cube:
            pending = [layer for layer in pending if layer != CUBE_LAYER]
        if pending:
            # The reader logs the cameras at the size the base stores; the lens videos are written re-encoded (write_lens_videos).
            reader: insta360_io.InsvReader = insta360_io.InsvReader(
                source.path,
                entity_path=schema.rig_path(RIG),
                timeline_name=schema.TIMELINE,
                frame_limit=self.config.frame_limit,
                max_lens_size=LENS_MAX_SIZE,
                videos=False,
            )
            self.write_layers(
                identity,
                targets,
                pending,
                {
                    paths.BASE_LAYER: lambda recording: self.write_base(recording, identity, reader),
                    PANORAMA_LAYER: lambda recording: self.write_panorama(recording, targets[paths.BASE_LAYER]),
                    CUBE_LAYER: lambda recording: self.write_cube(recording, targets[paths.BASE_LAYER]),
                },
                base_blueprint=segment_layout(panorama=PANORAMA_LAYER in pending),
            )
        return targets[paths.BASE_LAYER]

    def write_base(self, recording: rr.RecordingStream, identity: SequenceIdentity, reader: insta360_io.InsvReader) -> None:
        """The reader's chunks (cameras at the stored size, exposure, IMU), the lens videos re-encoded, then the exoego:v2
        tags, the IMU noise and the capture properties."""
        rig: insta360_io.InsvRig = reader.rig
        width, height = rig.resolution
        stored: tuple[int, int] = reader.stored_size
        for chunk in reader.chunks():
            rr.send_chunks(chunk, recording=recording)
        self.write_lens_videos(recording, rig, stored)
        profiled: bool = rig.profile_name is not None
        # Without a device profile the IMU has no pose in the rig (the reader logged the reason), so no sensor is the
        # rig frame and the IMU gets no noise model: slam-rs refuses such a rig instead of running VIO on it.
        log_rig_node(recording, RIG, reference="imu_00" if profiled else None, num_cameras=2, name=rig.camera_type)
        imu_node: str = schema.imu_path(RIG, 0)
        recording.log(imu_node, rr.AnyValues(name="imu", kind="imu"), static=True)
        noise: insta360_io.ImuNoise | None = rig.imu_noise
        if noise is not None:
            recording.log(
                imu_node,
                ImuCalibration(
                    gyro_noise_density=noise.gyro_noise_density,
                    accel_noise_density=noise.accel_noise_density,
                    gyro_bias_random_walk=noise.gyro_bias_random_walk,
                    accel_bias_random_walk=noise.accel_bias_random_walk,
                    rate_hz=noise.rate_hz,
                    source=noise.source,
                ),
                static=True,
            )
            recording.log(imu_node, rr.AnyValues(applied_time_shift_ns=0, time_shift_source="none: frames carry the profile's time model"), static=True)
        for cam in range(2):
            name: str = f"lens{cam}"
            recording.log(schema.cam_path(RIG, cam), rr.AnyValues(name=name, kind="rgb"), static=True)
            log_camera_source(
                recording,
                RIG,
                cam,
                name=name,
                source_width=width,
                source_height=height,
                video_codec="av1",
                cq=AV1_CQ,
                gop=AV1_GOP,
                stored_width=stored[0],
                stored_height=stored[1],
                stream_id=f"track{rig.lens_track[cam]}",
            )
        frame_ns: Int64[ndarray, " n"] = rig.frame_ns()
        self.timer.capture_s = float(frame_ns[-1] - frame_ns[0]) / 1e9
        writing.send_capture_properties(
            recording,
            identity,
            num_cameras=2,
            num_frames=rig.num_frames,
            start_time_ns=rig.start_ns,
            duration_s=self.timer.capture_s,
            clock_source="Insta360 trailer clock; "
            + ("frames at the middle row's mid-exposure time" if profiled else "frames at the raw exposure-record stamps (no device profile)"),
        )

    def write_lens_videos(self, recording: rr.RecordingStream, rig: insta360_io.InsvRig, stored: tuple[int, int]) -> None:
        """Store both lens videos box-filtered to ``stored`` (width, height) and encoded AV1 (``AV1_CQ``, a keyframe every
        ``AV1_GOP``), on the clip's frame times (``insta360_lens_video``).

        Lens ``k`` is read through the rig as lens ``k`` (whichever track holds it, its ``stream_id``) and becomes ``cam_0k``.
        On ``lens_device`` ``cuda`` no picture leaves the GPU: NVDEC (``decode_runs``), the box filter, NVENC from device
        memory (SVT-AV1 from host memory where NVENC does not encode AV1).
        """
        width, height = rig.resolution
        device: torch.device = layer_device(self.config.lens_device)
        with self.timer.stage("base:lenses"):
            pictures = lens_pictures(rig, np.arange(rig.num_frames, dtype=np.int64), device, batch=max(1, LENS_BATCH_PIXELS // (width * height)), threads=self.config.decode_threads)
            encode_and_log(
                recording,
                (torch.stack([area_downscale_uint8(lens, stored[::-1]) for lens in pair]) for pair in pictures),
                [schema.video_path(RIG, lens) for lens in range(2)],
                rig.frame_ns() - rig.start_ns,
                width=stored[0],
                height=stored[1],
                gop=AV1_GOP,
                cq=AV1_CQ,
                device=device,
            )

    def write_panorama(self, recording: rr.RecordingStream, base: Path) -> None:
        """Join every frameset's two lens images into the panorama (frame ``E``, module docstring) and log it as AV1 on the frame times.

        The tables come from the two cameras as the base recording states them (``cameras_from_recording``), the pictures
        from its lens videos (``BaseLensVideos``). On ``pano_device`` ``cuda`` no picture leaves the GPU: NVDEC, remap,
        flow seam, NVENC.
        """
        lenses: LensCameras = cameras_from_recording(base)
        source: BaseLensVideos = read_base_lens_videos(base)
        width: int = self.config.pano_width or insta360_panorama.default_panorama_width(lenses.size[0])
        table: insta360_panorama.PanoramaTable = insta360_panorama.panorama_table(lenses.cameras, lenses.lens1_from_lens0, lenses.size, width, self.config.pano_band_deg)
        width, height = table.size
        device: torch.device = layer_device(self.config.pano_device)
        with contextlib.closing(insta360_panorama.PanoramaJoin(table, lenses.size, device)) as join:
            rr.log(
                PANO_ENTITY,
                rr.AnyValues(
                    video_codec="av1",
                    cq=self.config.pano_cq,
                    gop=self.config.pano_gop,
                    width=width,
                    height=height,
                    seam=join.seam_kind,
                    seam_band_deg=self.config.pano_band_deg,
                ),
                static=True,
                recording=recording,
            )
            with self.timer.stage("panorama"):
                batch: int = max(1, LENS_BATCH_PIXELS // (lenses.size[0] * lenses.size[1]))
                pictures = lens_pictures(source, np.arange(source.num_frames, dtype=np.int64), device, batch=batch, threads=self.config.decode_threads)
                encode_and_log(
                    recording,
                    (join.panoramas(lens0, lens1)[None] for lens0, lens1 in pictures),
                    [PANO_ENTITY],
                    source.times_ns,
                    width=width,
                    height=height,
                    gop=self.config.pano_gop,
                    cq=self.config.pano_cq,
                    device=device,
                )

    def write_cube(self, recording: rr.RecordingStream, base: Path) -> None:
        """Join every sampled frameset's two lens images into the six cube faces and log each face as AV1 on the frameset's stamp.

        On ``cube_device``: both lenses decoded batch by batch, each face one remap of the stacked pair, six encoders fed from
        the device (``cuda``: no host copy of a picture).
        """
        lenses: LensCameras = cameras_from_recording(base)
        size: int = self.config.cube_face_size or cube.default_face_size(lenses.size[0])
        source: BaseLensVideos = read_base_lens_videos(base)
        indices: Int64[ndarray, " k"] = cube_indices(source.num_frames, self.config.cube_stride, self.config.cube_max_framesets)
        times_ns: Int64[ndarray, " k"] = source.times_ns[indices]
        faces: list[cube.CubeFace] = cube.cube_faces(size)
        tables: list[cube.FaceTable] = [cube.face_table(face, lenses.cameras, lenses.lens1_from_lens0, lenses.size) for face in faces]
        recording.send_property(
            "cube",
            rr.AnyValues(face_size=size, stride=self.config.cube_stride, framesets=int(indices.size), first_cam=FIRST_FACE_CAM, cq=self.config.cube_cq, gop=self.config.cube_gop),
        )
        for cam, face in zip(FACE_CAMS, faces, strict=True):
            node: str = schema.cam_path(RIG, cam)
            rr.log(
                node,
                rr.AnyValues(name=face.name, kind="rgb", camera_model="pinhole", video_codec="av1", cq=self.config.cube_cq, gop=self.config.cube_gop),
                static=True,
                recording=recording,
            )
            face_T_rig: Float64[ndarray, "4 4"] = cube.face_from_rig(face, lenses.lens_from_rig)
            rr.log(
                node,
                rr.Transform3D(translation=face_T_rig[:3, 3], mat3x3=face_T_rig[:3, :3], relation=rr.TransformRelation.ChildFromParent),
                static=True,
                recording=recording,
            )
            rr.log(
                schema.pinhole_path(RIG, cam),
                rr.Pinhole(image_from_camera=face.intrinsics, width=size, height=size, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.05),
                static=True,
                recording=recording,
            )
        device: torch.device = layer_device(self.config.cube_device)
        join = cube.CubeJoin(tables, lenses.size, device)
        with self.timer.stage("cube:faces"):
            batch: int = max(1, LENS_BATCH_PIXELS // (lenses.size[0] * lenses.size[1]))
            pictures = lens_pictures(source, indices, device, batch=batch, threads=self.config.decode_threads)
            encode_and_log(
                recording,
                (join.faces(lens0, lens1) for lens0, lens1 in pictures),
                [schema.video_path(RIG, cam) for cam in FACE_CAMS],
                times_ns,
                width=size,
                height=size,
                gop=self.config.cube_gop,
                cq=self.config.cube_cq,
                device=device,
            )

    def default_blueprint(self) -> rrb.Blueprint:
        """The base's layout: no panorama pane, since the panorama is an opt-in derived layer."""
        return segment_layout(panorama=False)

    def catalog_blueprints(self, entry: DatasetEntry) -> tuple[rrb.Blueprint, rrb.Blueprint]:
        """The segment layout shows the panorama tab when any registered segment holds the panorama layer, and its
        overviews frame the walk when the dataset holds one walked segment (``insta360_blueprint.segment_walk``)."""
        layer_names: list[list[str]] = entry.segment_table().to_arrow_table()["rerun_layer_names"].to_pylist()
        return segment_layout(panorama=any(PANORAMA_LAYER in names for names in layer_names), walk=segment_walk(entry)), self.table_blueprint()

    def table_blueprint(self) -> rrb.Blueprint:
        """Preview card: the rig beside lens 0."""
        return blueprints.table_blueprint(
            2,
            rig=RIG,
            run_source=RUN_SOURCE,
            eye_controls=blueprints.follow_eye_controls((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
            front_pane=blueprints.camera_view("lens0", RIG, 0, contents=[schema.video_path(RIG, 0)]),
        )
