"""Gen-HumanEgo ``depth`` layer: GenRobot's Ego-Depth on camera2, decoded from its PredZ codec into a ``DepthImage``.

``/robot0/sensor/camera2/depth`` ships one ``foxglove.CompressedImage`` per reference frame whose ``format`` is
``predz_i_zstd;dtype=float16``. ``/robot0/sensor/camera2/depth_info`` (one message) gives its size (512x416),
its own double-sphere lens (``K``, ``xi``, ``alpha``) and the codec, ``float16_predz_i_frame_zstd``. The codec,
read from GenRobot's ``das-datakit`` decoder: a zstd frame holding ``u32 mode_bytes, u32 size``, ``mode_bytes``
bytes the decoder skips, then one zigzag varint per pixel, row-major, the residual of the float16 bit pattern
against the prediction ``left + up - up_left`` (zero outside the image). That predictor inverts to a 2D prefix
sum, so a frame decodes with two ``cumsum`` calls. Values are metres along the optical axis (z-depth: the
datakit point cloud scales each double-sphere ray by ``depth / ray_z``); the fisheye border ships as NaN.

The depth image gets its own camera node, ``cam_06`` (``kind = "depth"``): it has its own lens and size, at
camera2's pose (the shipped depth is a resampling of camera2, ``frame_id = camera2_depth_predz``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Float64, Int64, UInt8, UInt16
from numpy import ndarray
from serde import serde
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters

from dataforge import schema
from dataforge.datasets.genrobot_layers import RIG, Retimer, shipped_counts
from dataforge.datasets.genrobot_source import REFERENCE_CAMERA, CameraCalibration, EpisodeClock, EpisodeFile, first_message, rig_T_cam
from dataforge.logging_toolkit import ChunkBatcher, log_camera_node

DEPTH_LAYER: str = "depth"
"""Metric Ego-Depth of camera2, one image per reference frame."""
DEPTH_TOPIC: str = f"/robot0/sensor/camera{REFERENCE_CAMERA}/depth"
"""PredZ-compressed float16 depth."""
DEPTH_INFO_TOPIC: str = f"/robot0/sensor/camera{REFERENCE_CAMERA}/depth_info"
"""One ``foxglove.DepthInfo``: size, lens and codec of the depth stream."""
CONFIDENCE_VIS_TOPIC: str = f"/robot0/sensor/camera{REFERENCE_CAMERA}/conf_vis_compressed"
"""H.264 picture of the depth confidence over the image; a rendering, not data."""
DEPTH_VIS_TOPIC: str = "/robot0/sensor/depth/vis_compressed"
"""H.264 colour-mapped picture of the depth; a rendering of ``DEPTH_TOPIC``."""
DEPTH_TOPICS: tuple[str, ...] = (DEPTH_TOPIC, CONFIDENCE_VIS_TOPIC, DEPTH_VIS_TOPIC)
"""Image topics kept out of the sidecar: the depth lives in its layer, the two renderings are not ingested."""
DEPTH_CAM: int = 6
"""Camera node of the depth image, after the six RGB cameras."""
CODEC: str = "float16_predz_i_frame_zstd"
"""The only depth codec dataforge decodes."""
IMAGE_PLANE_DISTANCE_M: float = 0.05
"""Frustum length, as the RGB cameras'."""
ZSTD_MAGIC: bytes = b"\x28\xb5\x2f\xfd"
"""Start of every zstd frame."""


@serde
@dataclass(frozen=True, slots=True)
class DepthInfo:
    """``foxglove.DepthInfo``; fields dataforge does not read (the per-frame stamps, the header) are allowed."""

    depth_topic: str
    """The topic it describes."""
    height: int
    """Decoded height, pixels."""
    width: int
    """Decoded width, pixels."""
    count: int
    """Depth frames in the episode."""
    K: list[float]
    """Intrinsics at the decoded size, 3x3 row-major."""
    xi: float
    """Double-sphere ``xi``."""
    alpha: float
    """Double-sphere ``alpha``."""
    codec: str
    """``float16_predz_i_frame_zstd``."""
    dtype: str
    """``float16``."""

    def __post_init__(self) -> None:
        if self.codec != CODEC or self.dtype != "float16":
            raise ValueError(f"unsupported depth codec {self.codec!r} ({self.dtype}); dataforge decodes {CODEC!r}")
        if len(self.K) != 9:
            raise ValueError(f"K has 9 values, got {len(self.K)}")

    @property
    def coefficients(self) -> list[float]:
        """``[fx, fy, cx, cy, xi, alpha]``, the order camera2's ``camera_info.D`` uses."""
        return [self.K[0], self.K[4], self.K[2], self.K[5], self.xi, self.alpha]


def zstd_content_size(frame: bytes) -> int:
    """The decompressed size a zstd frame header declares (RFC 8878, 3.1.1.1).

    Raises:
        ValueError: The bytes are not a zstd frame, or its header leaves the size out.
    """
    if frame[:4] != ZSTD_MAGIC:
        raise ValueError("not a zstd frame")
    descriptor: int = frame[4]
    size_flag: int = descriptor >> 6
    single_segment: bool = bool(descriptor & 0x20)
    if size_flag == 0 and not single_segment:
        raise ValueError("the zstd frame header has no content size")
    offset: int = 5 + (0 if single_segment else 1) + (0, 1, 2, 4)[descriptor & 0x03]
    width: int = (1, 2, 4, 8)[size_flag]
    size: int = int.from_bytes(frame[offset : offset + width], "little")
    return size + 256 if width == 2 else size


def decode_predz(frame: bytes, *, height: int, width: int) -> UInt8[ndarray, "n"]:
    """One PredZ depth frame as the little-endian float16 bytes of its ``height x width`` image, bit for bit.

    Raises:
        ValueError: The frame does not hold exactly one residual per pixel.
    """
    raw: UInt8[ndarray, "m"] = np.frombuffer(pa.Codec("zstd").decompress(frame, decompressed_size=zstd_content_size(frame)), dtype=np.uint8)
    mode_bytes, size = struct.unpack_from("<II", raw, 0)
    if mode_bytes != (height * width + 7) // 8:
        raise ValueError(f"{mode_bytes} mode bytes for a {width}x{height} image")
    body: UInt8[ndarray, "k"] = raw[8 + mode_bytes : 8 + size]
    ends: Int64[ndarray, "p"] = np.flatnonzero(body < 0x80)
    if ends.size != height * width or (body.size and ends[-1] != body.size - 1):
        raise ValueError(f"{ends.size} residuals for a {width}x{height} image")
    starts: Int64[ndarray, "p"] = np.concatenate([[0], ends[:-1] + 1])
    group: Int64[ndarray, "k"] = np.repeat(np.arange(ends.size), ends - starts + 1)
    shifts: Int64[ndarray, "k"] = (np.arange(body.size) - starts[group]) * 7
    values: Int64[ndarray, "p"] = np.zeros(ends.size, dtype=np.int64)
    np.add.at(values, group, (body & 0x7F).astype(np.int64) << shifts)
    residuals: Int64[ndarray, "p"] = np.where(values & 1, -((values + 1) >> 1), values >> 1)
    bits: UInt16[ndarray, "h w"] = (np.cumsum(np.cumsum(residuals.reshape(height, width), axis=0), axis=1) & 0xFFFF).astype("<u2")
    return bits.reshape(-1).view(np.uint8)


def depth_blobs(frames: pa.BinaryArray, *, height: int, width: int) -> pa.ListArray:
    """Each shipped frame as ``DepthImage`` buffer bytes; the NaN border becomes 0, the viewer's "no depth"."""
    images: list[UInt8[ndarray, "n"]] = []
    for frame in frames.to_pylist():
        depth: ndarray = decode_predz(frame, height=height, width=width).view(np.float16)
        images.append(np.where(np.isfinite(depth), depth, np.float16(0.0)).view(np.uint8))
    offsets: Int64[ndarray, "r"] = np.concatenate([[0], np.cumsum([image.size for image in images])]).astype(np.int32)
    values: UInt8[ndarray, "n"] = np.concatenate(images) if images else np.zeros(0, dtype=np.uint8)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(values))


def depth_path() -> str:
    """``/world/rig_00/cam_06/pinhole/depth``."""
    return schema.depth_path(RIG, DEPTH_CAM)


def depth_lens(info: DepthInfo) -> rrc.DeriveLens:
    """PredZ frames as ``DepthImage`` buffers; the format and the metre are static."""
    return rrc.DeriveLens("foxglove.CompressedImage:message", output_entity=depth_path()).to_component(
        rr.DepthImage.descriptor_buffer(), rrc.Selector(".data").pipe(lambda data: depth_blobs(data, height=info.height, width=info.width)), cast_to="auto"
    )


def depth_camera(info: DepthInfo, calibration: CameraCalibration) -> PinholeParameters:
    """The depth image's frustum: its own focal lengths and principal point, at camera2's pose."""
    transform: Float64[ndarray, "4 4"] = rig_T_cam(calibration.T_b_c)
    return PinholeParameters(
        name="camera2_depth",
        extrinsics=Extrinsics(world_R_cam=transform[:3, :3], world_t_cam=transform[:3, 3]),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF", fl_x=info.K[0], fl_y=info.K[4], cx=info.K[2], cy=info.K[5], height=info.height, width=info.width
        ),
    )


def read_depth_info(mcap: Path) -> DepthInfo:
    """The episode's one ``depth_info`` message."""
    return first_message(EpisodeFile(mcap), DEPTH_INFO_TOPIC, DepthInfo)


def write_depth_layer(recording: rr.RecordingStream, mcap: Path, *, clock: EpisodeClock, calibration: CameraCalibration, info: DepthInfo) -> int:
    """The ``depth`` layer: the depth camera node and one ``DepthImage`` per shipped frame, on the episode clock.

    Returns:
        Depth frames written.

    Raises:
        ValueError: The written frames differ from the MCAP's message count or from ``depth_info.count``.
    """
    log_camera_node(
        recording, RIG, DEPTH_CAM, depth_camera(info, calibration),
        name="camera2_depth", kind="depth", image_plane_distance=IMAGE_PLANE_DISTANCE_M, camera_model="ds",
    )  # fmt: skip
    rr.log(
        schema.cam_path(RIG, DEPTH_CAM),
        rr.AnyValues(ds_coefficients=pa.array([info.coefficients], type=pa.list_(pa.float64())), codec=info.codec, source_topic=DEPTH_TOPIC),
        static=True,
        recording=recording,
    )
    rr.log(
        depth_path(),
        rr.DepthImage.from_fields(
            format=rr.components.ImageFormat(width=info.width, height=info.height, channel_datatype=rr.ChannelDatatype.F16), meter=1.0, colormap="turbo"
        ),
        static=True,
        recording=recording,
    )
    retimer: Retimer = Retimer(clock, native_rate=frozenset())
    batcher: ChunkBatcher = ChunkBatcher()
    stream: rrc.LazyChunkStream = EpisodeFile(mcap).stream([DEPTH_TOPIC]).flat_map(retimer).lenses(depth_lens(info)).flat_map(batcher)
    recording.send_chunks(stream)
    recording.send_chunks(batcher.flush())
    written: int = retimer.rows[DEPTH_TOPIC]
    shipped: int = shipped_counts(mcap).get(DEPTH_TOPIC, 0)
    if written != shipped or written != info.count:
        raise ValueError(f"{mcap}: wrote {written} depth frames, the MCAP holds {shipped} and depth_info counts {info.count}")
    recording.send_property(
        DEPTH_LAYER,
        rr.AnyValues(
            source_topic=pa.array([DEPTH_TOPIC], type=pa.string()),
            codec=pa.array([info.codec], type=pa.string()),
            num_frames=pa.array([written], type=pa.int64()),
            value=pa.array(["z-depth in metres along camera2's optical axis; the shipped NaN border is 0"], type=pa.string()),
        ),
    )
    return written
