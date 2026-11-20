"""Insta360 dual-fisheye ``.insv`` clips: one base recording per clip, and on request a derived panorama layer.

The base is ``insta360_io.InsvReader``'s chunk stream (packages/insta360-io; the reader's module docstring states its
entities, the device profile and the clock), sent as it comes with ``entity_path`` the rig ``/world/rig_00`` and
``timeline_name`` ``video_time`` (``capture:start_time_ns`` is its zero). ``cam_00`` and ``cam_01`` are lens 0 and lens
1, whichever track holds them (``stream_id``). This module adds only the exoego:v2 tags (rig node, sensor names, camera
sources), the IMU noise model and the capture properties. A clip without a device profile gets no rig reference and no
IMU noise model, so slam-rs refuses it. The IMU noise model is a placeholder, marked as such in its ``source``.

Panorama layer (``--panorama``; off by default). ``panorama`` holds the 1920x960 equirectangular panorama at
``/world/rig_00/pano``, stitched on the CPU from the two lens images of every frameset and encoded AV1 on NVENC, on
the frame times. It is a derived output: nothing reads it as input, slam-rs tracks the two lens cameras of the base,
and the lens brightness steps at the seams are not corrected. The stitch reads the two cameras from the base
recording as written (float32 statics), so a later ``--panorama`` run adds the layer beside an existing base without
writing the base again.

Panorama frame ``E``. Axes = camera 0's (RDF), origin = the midpoint of the two camera centres. Pixel ``(u, v)``
(pixel centres at integers) looks along longitude ``lon = (u + 0.5) pi / 960 - pi / 2``, ``lon = atan2(x, z)`` in
``E``, and latitude ``lat = (v + 0.5) pi / 960 - pi / 2``, ``+y`` down: the left half is camera 0's hemisphere (centre
column 480 = camera 0's ``+Z``), the right half camera 1's (column 1440 = ``-Z``). Each half is stitched from its own
camera centre (only camera 1's rotation in camera 0 enters the tables), so the panorama ignores the lens-centre
baseline: 33.3 mm on the X4 Air, 31.9 mm on the X5.

From the native poses to the panorama. ``E`` is fixed in the rig: ``T_cam0_E = (I, c1 / 2)``, with ``c1`` camera 1's
centre in camera 0 (the translation of ``cam0_T_cam1``, from the ``cam_00`` and ``cam_01`` ``Transform3D``s,
``ChildFromParent``, as ``stitch_from_recording`` reads them). A COLMAP ``EQUIRECTANGULAR`` camera ``C`` (single
centre; ``u = W (lon_C + pi) / (2 pi)`` in pixel-corner coordinates, image centre = ``+Z_C``, ``+y`` down) sees this
panorama with its columns a quarter turn on, ``lon_E = lon_C + pi / 2``, so ``T_cam0_C = (Ry(+90 deg), c1 / 2)`` with
``Ry(+90 deg) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]`` (``+Z_C`` is camera 0's ``+X``). slam-rs reports ``world_from_rig``
in the IMU frame ``imu_00`` (the rig reference), so a tracked pose ``T_world_imu`` becomes
``T_world_C = T_world_imu * T_imu_cam0 * T_cam0_C`` (COLMAP's ``cam_from_world`` is its inverse), with ``T_imu_cam0``
the inverse of the ``cam_00`` ``Transform3D``. Every factor is a static of the base, so the transform needs no file of
its own; with the profiles' zero lever arm the midpoint, and so the panorama centre, is the IMU origin.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import av
import insta360_io
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.imu_calibration import ImuCalibration

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import log_camera_source, log_rig_node, log_video_stream
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4, work_dir

RIG: int = 0
"""One rig per clip."""
PANORAMA_LAYER: str = "panorama"
"""The derived panorama, written only under ``--panorama``: an AV1 ``VideoStream`` at ``PANO_ENTITY`` on the frame times."""
PANO_ENTITY: str = f"{schema.rig_path(RIG)}/pano"
"""Where the panorama layer puts its video."""
PANORAMA_SIZE: tuple[int, int] = (1920, 960)
"""Width and height of ``insta360_io.PanoramaStitch.panorama``."""
RUN_SOURCE: str = "slam_rs"
"""The run layer the layouts show: slam-rs tracks these clips."""


def gray_frames(rig: insta360_io.InsvRig, lens: int, *, threads: int) -> Iterator[UInt8[ndarray, "h w"]]:
    """Every picture of one lens as C-contiguous ``gray8``, read one GOP at a time.

    ``reformat(format="gray8")`` as slam-rs's feed decodes, so the panorama is stitched from the pixels the tracker sees.
    """
    decoder = av.CodecContext.create(rig.codec, "r")
    decoder.thread_count = threads
    decoder.thread_type = "NONE" if threads == 1 else "FRAME"

    def gray(frame: av.VideoFrame) -> UInt8[ndarray, "h w"]:
        converted: av.VideoFrame = frame.reformat(format="gray8")
        plane = converted.planes[0]
        padded: UInt8[ndarray, "h stride"] = np.frombuffer(plane, dtype=np.uint8).reshape(converted.height, plane.line_size)
        return padded[:, : converted.width].copy()

    for start, stop in rig.gops(lens).tolist():
        data, offsets, _ = rig.read_packed(lens, start, stop)
        for first, last in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
            yield from (gray(frame) for frame in decoder.decode(av.Packet(data[first:last].tobytes())))
    yield from (gray(frame) for frame in decoder.decode(None))


def stitch_from_recording(base: Path) -> insta360_io.PanoramaStitch:
    """The panorama stitch of a base recording's ``cam_00`` and ``cam_01``, from their statics as written (float32).

    Each camera is its ``DistortionModel``, ``DistortionCoefficients`` and pinhole matrix; camera 1's pose in camera 0
    is ``cam0_T_rig * cam1_T_rig^-1`` from the two ``Transform3D``s (``ChildFromParent``).

    Raises:
        ValueError: If a camera is not a Mei camera or its statics are missing.
    """
    entities: list[str] = [schema.cam_path(RIG, cam) for cam in range(2)] + [schema.pinhole_path(RIG, cam) for cam in range(2)]
    statics: dict[str, object] = {}
    for chunk in rrc.RrdReader(base).stream():
        if chunk.is_static and chunk.entity_path in entities:
            batch = chunk.to_record_batch()
            statics.update({f"{chunk.entity_path}:{name}": batch.column(name).to_pylist()[0] for name in batch.schema.names if not name.startswith("rerun")})
    cameras: list[insta360_io.MeiCamera] = []
    cam_T_rig: list[Float64[ndarray, "4 4"]] = []
    for cam in range(2):
        pinhole: str = schema.pinhole_path(RIG, cam)
        node: str = schema.cam_path(RIG, cam)
        missing: list[str] = [
            key
            for key in (f"{pinhole}:simplecv.components.DistortionModel", f"{pinhole}:simplecv.components.DistortionCoefficients", f"{pinhole}:Pinhole:image_from_camera", f"{node}:Transform3D:mat3x3", f"{node}:Transform3D:translation")
            if not statics.get(key)
        ]
        if missing:
            raise ValueError(f"{base}: camera {cam} lacks {missing}")
        model: str = str(statics[f"{pinhole}:simplecv.components.DistortionModel"][0])  # pyrefly: ignore[bad-index]
        coefficients: list[float] = list(statics[f"{pinhole}:simplecv.components.DistortionCoefficients"][0])  # pyrefly: ignore[bad-index]
        # Rerun stores the 3x3 matrices column-major.
        k: Float64[ndarray, "3 3"] = np.asarray(statics[f"{pinhole}:Pinhole:image_from_camera"][0], dtype=np.float64).reshape(3, 3).T  # pyrefly: ignore[bad-index]
        cameras.append(insta360_io.MeiCamera(model, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), coefficients))
        pose: Float64[ndarray, "4 4"] = np.eye(4)
        pose[:3, :3] = np.asarray(statics[f"{node}:Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T  # pyrefly: ignore[bad-index]
        pose[:3, 3] = np.asarray(statics[f"{node}:Transform3D:translation"][0], dtype=np.float64)  # pyrefly: ignore[bad-index]
        cam_T_rig.append(pose)
    cam0_T_cam1: Float64[ndarray, "4 4"] = cam_T_rig[0] @ np.linalg.inv(cam_T_rig[1])
    width, height = (int(v) for v in statics[f"{schema.pinhole_path(RIG, 0)}:Pinhole:resolution"][0])  # pyrefly: ignore[bad-index]
    return insta360_io.PanoramaStitch(cameras[0], cameras[1], cam0_T_cam1.tolist(), (width, height))


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
    """Also write the panorama layer (CPU stitch + AV1 NVENC, about 10x the base's time). Run again with it to add the
    layer beside bases already written."""
    decode_threads: int = 4
    """PyAV threads per lens decoder of the panorama layer."""
    pano_cq: int = 34
    """NVENC constant-quality target of the panorama: a viewing layer, not an analysis input."""
    pano_gop: int = 30
    """Panorama keyframe interval in frames, about one second."""


class Insta360Dataset(DataforgeDataset[Insta360Config, ClipSource]):
    """Converts ``.insv`` clips into a base layer (lens videos, IMU, calibration) and, on request, a panorama layer."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, PANORAMA_LAYER)

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
        if pending:
            reader: insta360_io.InsvReader = insta360_io.InsvReader(
                source.path, entity_path=schema.rig_path(RIG), timeline_name=schema.TIMELINE, frame_limit=self.config.frame_limit
            )
            self.write_layers(
                identity,
                targets,
                pending,
                {
                    paths.BASE_LAYER: lambda recording: self.write_base(recording, identity, reader),
                    PANORAMA_LAYER: lambda recording: self.write_panorama(recording, reader.rig, targets[paths.BASE_LAYER]),
                },
                base_blueprint=self.segment_blueprint(panorama=PANORAMA_LAYER in pending),
            )
        return targets[paths.BASE_LAYER]

    def write_base(self, recording: rr.RecordingStream, identity: SequenceIdentity, reader: insta360_io.InsvReader) -> None:
        """The reader's chunks (videos, cameras, exposure, IMU), then the exoego:v2 tags, the IMU noise and the capture properties."""
        rig: insta360_io.InsvRig = reader.rig
        width, height = rig.resolution
        for chunk in reader.chunks():
            rr.send_chunks(chunk, recording=recording)
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
            log_camera_source(recording, RIG, cam, name=name, source_width=width, source_height=height, video_codec=rig.codec, stream_id=f"track{rig.lens_track[cam]}")
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

    def write_panorama(self, recording: rr.RecordingStream, rig: insta360_io.InsvRig, base: Path) -> None:
        """Stitch every frameset's two lens images into the panorama (frame ``E``, module docstring) and log it as AV1 on the frame times.

        The stitch is built from the two cameras as the base recording states them (``stitch_from_recording``), so it
        is the stitch of anything else that reads that recording.
        """
        stitch: insta360_io.PanoramaStitch = stitch_from_recording(base)
        lens0, lens1 = (gray_frames(rig, lens, threads=self.config.decode_threads) for lens in range(2))
        panoramas: Iterator[bytes] = (stitch.panorama(a, b).tobytes() for a, b in zip(lens0, lens1, strict=True))
        width, height = PANORAMA_SIZE
        with work_dir("insta360-pano-") as work:
            video: Path = work / "pano.mp4"
            with self.timer.stage("encode"):
                # fps is the container's nominal rate only; log_video_stream puts every sample on its frame time.
                encode_frames_to_mp4(
                    panoramas, video, source=FrameSource("gray8", width=width, height=height), fps=30, gop=self.config.pano_gop, cq=self.config.pano_cq
                )
            rr.log(PANO_ENTITY, rr.AnyValues(video_codec="av1", cq=self.config.pano_cq, gop=self.config.pano_gop), static=True, recording=recording)
            samples: int = log_video_stream(recording, video, PANO_ENTITY, times_ns=rig.frame_ns() - rig.start_ns)
        if samples != rig.num_frames:
            raise ValueError(f"the panorama holds {samples} frames for {rig.num_frames} framesets")

    def default_blueprint(self) -> rrb.Blueprint:
        """The base's layout: no panorama pane, since the panorama is an opt-in derived layer."""
        return self.segment_blueprint(panorama=False)

    def catalog_blueprints(self, entry: DatasetEntry) -> tuple[rrb.Blueprint, rrb.Blueprint]:
        """The segment layout shows the panorama pane when any registered segment holds the panorama layer."""
        layer_names: list[list[str]] = entry.segment_table().to_arrow_table()["rerun_layer_names"].to_pylist()
        return self.segment_blueprint(panorama=any(PANORAMA_LAYER in names for names in layer_names)), self.table_blueprint()

    def segment_blueprint(self, *, panorama: bool) -> rrb.Blueprint:
        """Both lenses (and with ``panorama`` the panorama) beside the rig in 3D, over the IMU plots."""
        lenses: list[rrb.Spatial2DView] = [blueprints.camera_view(f"lens{cam}", RIG, cam, contents=[schema.video_path(RIG, cam)]) for cam in range(2)]
        return blueprints.rig_blueprint(
            lenses + ([rrb.Spatial2DView(name="panorama", origin=PANO_ENTITY)] if panorama else []),
            rig=RIG,
            run_source=RUN_SOURCE,
            # Lens 0 looks along z_P = -y_imu; image up is -y_P = -x_imu.
            eye_controls=blueprints.follow_eye_controls((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
            plots=blueprints.imu_plots(RIG, 0),
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Preview card: the rig beside lens 0."""
        return blueprints.table_blueprint(
            2,
            rig=RIG,
            run_source=RUN_SOURCE,
            eye_controls=blueprints.follow_eye_controls((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
            front_pane=blueprints.camera_view("lens0", RIG, 0, contents=[schema.video_path(RIG, 0)]),
        )
