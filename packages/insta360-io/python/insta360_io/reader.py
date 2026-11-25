"""``InsvReader``: an Insta360 ``.insv`` clip as Rerun chunks, in the layout of ``rerun.experimental.Mp4Reader``.

Rerun 0.42's ``Mp4Reader`` cannot select a track, and a clip holds two (one per lens), so this reader emits the
same chunk shapes itself, per lens: one static ``VideoStream:codec`` chunk, one ``VideoStream:sample`` chunk per GOP
(a keyframe and the samples up to the next one) on the ``timeline_name`` duration timeline, and one trailing
``VideoStream:is_keyframe`` chunk with a ``True`` row per keyframe. The samples are the tracks' own H.264 or HEVC
bitstream (``VideoStream`` codec ``H264`` or ``H265``), remuxed to Annex B with the parameter sets in front of every
keyframe (H.264 IDR, HEVC IRAP), never re-encoded. Rust (``InsvRig``) reads and packs them without the GIL; the arrays
reach Rerun without a copy. Two-file clips (ONE X2, ONE R/RS, X3: lens 0 in ``…_00_…``, lens 1 in ``…_10_…``) open from
either member.

Beside the videos the reader states what the clip and its device profile say, as one camera rig under
``entity_path``, rig frame = IMU frame:

- ``cam_NN``: static ``Transform3D`` ``cam_from_rig`` (``ChildFromParent``) and ``AnyValues`` ``camera_model`` and
  ``rolling_shutter_readout_s`` (first row to last; row ``v`` of ``H`` is read ``readout * (v - (H - 1) / 2) / H``
  after the frame time; absent when neither the clip nor a profile states it).
- ``cam_NN/pinhole``: static ``Pinhole`` (the paraxial ``gamma / (1 + xi)``, ``cx, cy``, track pixels, RDF: a viewer's
  pinhole approximation of the lens, no calibration) and the camera model as the monorepo's
  ``simplecv.components.DistortionModel`` (``mei_v6_unified``, or ``mei_v3_unified`` without a V6 lens string) and
  ``simplecv.components.DistortionCoefficients`` (``gamma_x, gamma_y, xi, half_field_rad, d0 .. d12``; ``mei_v3``:
  ``... k1, k2, k3, p1, p2``): exoego:v2 section 4, simplecv's ``MeiParameters``, which slam-rs, dataforge and posekit
  read. Bases written before the unified names (``mei_v6`` / ``mei_v3``: the coefficients without gamma, gamma as the
  ``Pinhole`` focal) still read the same.
- ``cam_NN/pinhole/video``: the lens video, as above.
- ``cam_NN/exposure``: per-frame exposure time (s), record 4 for lens 0 and record 12 for lens 1.
- ``imu_00``: static identity ``Transform3D`` (the IMU is the rig origin; absent without a device profile); ``imu_00/gyro`` (rad/s) and
  ``imu_00/accel`` (m/s^2) as ``Scalars`` in the IMU's axes.
- the recording property ``insta360``: camera type, firmware, device profile and its version, ``pano_from_imu``.

Without a device profile. A body nobody has measured, or a clip outside every measured mode of its body, is still read,
with one warning naming the body and the reason: rig frame = the vendor lens frame P (the lens poses are the clip's own
factory calibration), frames at the raw exposure-record stamps, no ``Transform3D`` at ``imu_00`` (the IMU has no pose
in the rig), and the ``insta360`` property states ``profile = "none"`` and ``profile_note`` (the reason) instead of the
version and ``pano_from_imu``. Visual-inertial consumers (slam-rs) refuse such a recording.

Stored size. A consumer that stores the lens videos re-encoded at a smaller size (dataforge's Insta360 base) asks for
``max_lens_size``: the lenses are then halved until they are at most that wide (``lens_halvings``) and each camera is
logged for the halved image (``halved_camera``: a box filter puts camera pixel ``p`` at ``(p + 0.5) s - 0.5``,
``s = 2^-halvings``), with the halved ``Pinhole`` size. ``videos=False`` leaves out the lens videos (their codec, sample
and keyframe chunks), for that consumer to write its own.

Time. Every temporal chunk is on one duration timeline (``timeline_name``, ``video`` by default as in
``Mp4Reader``): nanoseconds since the earliest frame or IMU sample on the trailer clock (``InsvRig.start_ns``).
Frames are at their effective time on the IMU clock (the device profile's frame-time model), not at the mp4 PTS.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float64, Int32, Int64, UInt8
from numpy import ndarray
from rerun.chunk import Chunk, LazyChunkStream

from insta360_io._insta360_io import InsvRig, MeiCamera

LOG: logging.Logger = logging.getLogger(__name__)
NO_PROFILE: str = "none"
"""The ``insta360`` property's ``profile`` of a clip read without a device profile."""
CODECS: dict[str, rr.VideoCodec] = {"h264": rr.VideoCodec.H264, "hevc": rr.VideoCodec.H265}
"""``InsvRig.codec`` (FFmpeg's names) as Rerun's codec."""

ROWS_PER_CHUNK: int = 65_536
"""IMU and exposure samples per chunk (about a minute of the 1 kHz IMU)."""
DISTORTION_MODEL: rr.ComponentDescriptor = rr.ComponentDescriptor("simplecv.components.DistortionModel", component_type="simplecv.components.DistortionModel")
"""The camera model name, as ``simplecv.rerun_custom_types.CameraDistortion`` writes it."""
DISTORTION_COEFFICIENTS: rr.ComponentDescriptor = rr.ComponentDescriptor(
    "simplecv.components.DistortionCoefficients", component_type="simplecv.components.DistortionCoefficients"
)
"""The camera model coefficients (float32 list), as ``CameraDistortion`` writes them."""
BLOB: pa.DataType = pa.list_(pa.field("item", pa.uint8(), nullable=False))
"""Arrow type of one ``VideoSample``."""


def lens_halvings(width: int, max_lens_size: int | None) -> int:
    """How many times a lens ``width`` pixels wide is halved to be at most ``max_lens_size`` wide (0 with None).

    Raises:
        ValueError: If ``max_lens_size`` is not positive (no halving reaches it).
    """
    if max_lens_size is not None and max_lens_size <= 0:
        raise ValueError(f"max_lens_size must be positive, got {max_lens_size}")
    count: int = 0
    while max_lens_size is not None and (width >> count) > max_lens_size:
        count += 1
    return count


def halved_camera(camera: MeiCamera, halvings: int) -> MeiCamera:
    """``camera`` on its image box-filtered ``halvings`` times: focal ``f s``, principal point ``(c + 0.5) s - 0.5``,
    ``s = 2^-halvings``; the unitless coefficients (``xi``, the field and the distortion terms) are unchanged.

    The Mei projection is ``gamma m(x) + c`` with ``m`` unitless, so the halved camera projects every ray onto
    ``(p + 0.5) s - 0.5`` of the full camera's ``p``: the stored pixel whose box holds it.
    """
    scale: float = 1.0 / (1 << halvings)
    return MeiCamera(camera.model, camera.fx * scale, camera.fy * scale, (camera.cx + 0.5) * scale - 0.5, (camera.cy + 0.5) * scale - 0.5, list(camera.coefficients))


@dataclass(frozen=True, slots=True)
class Described:
    """One component batch: a descriptor and its Arrow array (one row per instance)."""

    descriptor: rr.ComponentDescriptor
    array: pa.Array

    def component_descriptor(self) -> rr.ComponentDescriptor:
        return self.descriptor

    def as_arrow_array(self) -> pa.Array:
        return self.array


def described(*archetypes: rr.AsComponents) -> list[Described]:
    """The component batches of archetypes."""
    return [Described(b.component_descriptor(), b.as_arrow_array()) for a in archetypes for b in a.as_component_batches()]


def static_chunk(entity_path: str, batches: list[Described]) -> Chunk:
    """One static row holding every batch whole."""
    columns = [rr.ComponentColumn(b.descriptor, b, lengths=[len(b.array)]) for b in batches]
    return Chunk.from_columns(entity_path, [], columns)


def row_column(descriptor: rr.ComponentDescriptor, array: pa.Array) -> rr.ComponentColumn:
    """One component per row."""
    return rr.ComponentColumn(descriptor, Described(descriptor, array))


def blobs(data: UInt8[ndarray, " n_bytes"], offsets: Int32[ndarray, " n_plus_1"]) -> pa.Array:
    """Packed samples as Arrow blobs, sharing the arrays' memory."""
    values: pa.Array = pa.Array.from_buffers(pa.uint8(), int(data.size), [None, pa.py_buffer(data)])
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values, type=BLOB)


class InsvReader:
    """Read chunks from an Insta360 ``.insv`` clip: both lens videos, calibration, IMU and exposure (module docstring)."""

    def __init__(
        self,
        path: str | Path,
        *,
        entity_path: str | None = None,
        timeline_name: str = "video",
        frame_limit: int | None = None,
        max_lens_size: int | None = None,
        videos: bool = True,
    ) -> None:
        """Open the clip and select its device profile.

        Args:
            path: The ``.insv`` clip.
            entity_path: Root of the rig; like ``Mp4Reader``, the absolute file path when ``None``.
            timeline_name: The duration timeline every temporal chunk is on.
            frame_limit: Keep the first frames only (and the IMU up to 1 s past them).
            max_lens_size: Log the cameras for the lenses halved until they are at most this wide (module docstring,
                "Stored size"); None: the track size.
            videos: Emit the lens videos; False leaves them out for the caller to write.

        Raises:
            ValueError: On a malformed clip, a non-positive ``max_lens_size``, or ``videos`` with halved cameras (the
                track-size videos would not match the cameras).
        """
        self.path: Path = Path(path).absolute()
        self.entity_path: str = str(self.path) if entity_path is None else entity_path
        self.timeline_name: str = timeline_name
        self.rig: InsvRig = InsvRig(str(self.path), frame_limit)
        self.videos: bool = videos
        self.halvings: int = lens_halvings(self.rig.resolution[0], max_lens_size)
        """Times each lens camera is halved (``lens_halvings``)."""
        if videos and self.halvings:
            raise ValueError(f"{self.path.name}: cameras halved {self.halvings} times describe re-encoded videos, not the track's: pass videos=False")
        if self.rig.profile_name is None:
            LOG.warning("%s (%s): read without a device profile: %s", self.path.name, self.rig.camera_type, self.rig.profile_note)

    def __repr__(self) -> str:
        return f"InsvReader({self.path})"

    def stream(self) -> LazyChunkStream:
        """A lazy stream over every chunk: statics, IMU and exposure, then the GOPs of both lenses, then keyframes."""
        return LazyChunkStream.from_iter(self.chunks())

    def cam(self, lens: int) -> str:
        """Entity of one lens camera."""
        return f"{self.entity_path}/cam_{lens:02d}"

    def video_entity(self, lens: int) -> str:
        """Entity of one lens video."""
        return f"{self.cam(lens)}/pinhole/video"

    @property
    def stored_size(self) -> tuple[int, int]:
        """Width and height of the lens images the cameras are logged for."""
        width, height = self.rig.resolution
        return width >> self.halvings, height >> self.halvings

    def time(self, times_ns: Int64[ndarray, " n"]) -> rr.TimeColumn:
        """Times on the trailer clock as the reader's duration timeline."""
        return rr.TimeColumn(self.timeline_name, duration=(times_ns - self.rig.start_ns).view("timedelta64[ns]"))

    def chunks(self) -> Iterator[Chunk]:
        """Every chunk, in stream order."""
        rig: InsvRig = self.rig
        yield from self.static_chunks()
        imu_ns, accel, gyro = rig.imu()
        for first in range(0, imu_ns.size, ROWS_PER_CHUNK):
            window = slice(first, first + ROWS_PER_CHUNK)
            for name, values in (("gyro", gyro), ("accel", accel)):
                yield Chunk.from_columns(f"{self.entity_path}/imu_00/{name}", [self.time(imu_ns[window])], rr.Scalars.columns(scalars=values[window]))
        frame_ns: Int64[ndarray, " n"] = rig.frame_ns()
        exposure: Float64[ndarray, "2 n"] = rig.exposure_s()
        for lens in range(2):
            for first in range(0, frame_ns.size, ROWS_PER_CHUNK):
                window = slice(first, first + ROWS_PER_CHUNK)
                yield Chunk.from_columns(f"{self.cam(lens)}/exposure", [self.time(frame_ns[window])], rr.Scalars.columns(scalars=exposure[lens, window]))
        if not self.videos:
            return
        gops: list[Int64[ndarray, "g 2"]] = [rig.gops(lens) for lens in range(2)]
        keyframe_ns: list[list[Int64[ndarray, " k"]]] = [[], []]
        # Lens 0 and lens 1 GOP by GOP, so a consumer that stops early holds both lenses up to the same time.
        for index in range(max(len(g) for g in gops)):
            for lens in range(2):
                if index < len(gops[lens]):
                    start, stop = (int(v) for v in gops[lens][index])
                    data, offsets, keyframes = rig.read_packed(lens, start, stop)
                    times: Int64[ndarray, " n"] = frame_ns[start:stop]
                    keyframe_ns[lens].append(times[keyframes])
                    yield Chunk.from_columns(self.video_entity(lens), [self.time(times)], [row_column(rr.VideoStream.descriptor_sample(), blobs(data, offsets))])
        for lens in range(2):
            times = np.concatenate(keyframe_ns[lens]) if keyframe_ns[lens] else np.zeros(0, dtype=np.int64)
            flags: Bool[ndarray, " k"] = np.ones(times.size, dtype=np.bool_)
            yield Chunk.from_columns(self.video_entity(lens), [self.time(times)], [row_column(rr.VideoStream.descriptor_is_keyframe(), pa.array(flags))])

    def static_chunks(self) -> Iterator[Chunk]:
        """The rig's statics: per lens the pose, camera, codec and series names; the IMU node; the clip property."""
        rig: InsvRig = self.rig
        width, height = self.stored_size
        for lens, factory in enumerate(rig.lenses):
            camera: MeiCamera = halved_camera(factory.camera, self.halvings)
            rig_T_lens: Float64[ndarray, "4 4"] = np.asarray(rig.rig_from_lens(lens), dtype=np.float64)
            lens_R_rig: Float64[ndarray, "3 3"] = rig_T_lens[:3, :3].T
            yield static_chunk(
                self.cam(lens),
                described(
                    rr.Transform3D(translation=-lens_R_rig @ rig_T_lens[:3, 3], mat3x3=lens_R_rig, relation=rr.TransformRelation.ChildFromParent),
                    rr.AnyValues(camera_model=camera.model, rolling_shutter_readout_s=rig.readout_s),
                ),
            )
            # exoego:v2 section 4: the Mei model's own focal (gamma) goes in the coefficients under the unified name; the
            # Pinhole holds the paraxial gamma / (1 + xi), a viewer's pinhole approximation of the lens, no calibration.
            xi: float = float(camera.coefficients[0])
            pinhole = rr.Pinhole(
                image_from_camera=[[camera.fx / (1.0 + xi), 0.0, camera.cx], [0.0, camera.fy / (1.0 + xi), camera.cy], [0.0, 0.0, 1.0]],
                width=width,
                height=height,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.05,
            )
            unified: list[float] = [camera.fx, camera.fy, *camera.coefficients]
            coefficients: pa.Array = pa.array([np.asarray(unified, dtype=np.float32)], type=pa.list_(pa.float32()))
            distortion: list[Described] = [
                Described(DISTORTION_MODEL, pa.array([f"{camera.model}_unified"])),
                Described(DISTORTION_COEFFICIENTS, coefficients),
            ]
            yield static_chunk(f"{self.cam(lens)}/pinhole", described(pinhole) + distortion)
            if self.videos:
                yield static_chunk(self.video_entity(lens), described(rr.VideoStream(codec=CODECS[rig.codec])))
            yield static_chunk(f"{self.cam(lens)}/exposure", described(rr.SeriesLines(names=["exposure_s"])))
        clip = {"source_file": self.path.name, "camera_type": rig.camera_type, "firmware": rig.firmware}
        if rig.profile_name is None:
            yield Chunk.from_property("insta360", rr.AnyValues(**clip, profile=NO_PROFILE, profile_note=rig.profile_note))
            return
        # The IMU is the rig origin.
        yield static_chunk(f"{self.entity_path}/imu_00", described(rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3, dtype=np.float32))))
        yield Chunk.from_property(
            "insta360",
            rr.AnyValues(
                **clip,
                profile=rig.profile_name,
                profile_version=rig.profile_version,
                pano_from_imu=np.asarray(rig.pano_from_imu, dtype=np.float64).ravel().tolist(),
            ),
        )
