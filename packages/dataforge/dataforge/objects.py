"""Shared sparse object poses and confidence-driven static mesh visibility."""

import json
import struct

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import schema
from dataforge.logging_toolkit import frame_index_column, time_column


def log_object_pose(
    recording: rr.RecordingStream,
    alias: str,
    times_ns: Int64[ndarray, "t"],
    frame_indices: Int64[ndarray, "t"],
    transforms: Float64[ndarray, "t 4 4"],
    confidence: Float32[ndarray, "t"] | Float64[ndarray, "t"],
    *,
    trust_threshold: float,
) -> None:
    """Write trusted Float64[t,4,4] world transforms (metres), and all Float32/Float64[t] confidences.

    Missing transforms are NaN. Trust is strictly above the supplied threshold.
    """
    valid: Bool[ndarray, "t"] = np.isfinite(transforms).all(axis=(1, 2)) & (confidence > trust_threshold)
    if np.any(valid):
        rr.send_columns(
            schema.objects_path(alias),
            indexes=[time_column(times_ns[valid]), frame_index_column(frame_indices[valid])],
            columns=rr.Transform3D.columns(
                translation=transforms[valid, :3, 3], quaternion=Rotation.from_matrix(transforms[valid, :3, :3]).as_quat()
            ),
            recording=recording,
        )
    rr.send_columns(
        schema.object_confidence_path(alias),
        indexes=[time_column(times_ns), frame_index_column(frame_indices)],
        columns=rr.Scalars.columns(scalars=confidence),
        recording=recording,
    )


def log_object_mesh(
    recording: rr.RecordingStream,
    alias: str,
    times_ns: Int64[ndarray, "t"],
    frame_indices: Int64[ndarray, "t"],
    asset: rr.Asset3D,
    confidence: Float32[ndarray, "t"] | Float64[ndarray, "t"],
    *,
    trust_threshold: float,
) -> None:
    """Log static geometry and alpha transitions from Float32/Float64[t] source confidence."""
    path: str = schema.object_mesh_path(alias)
    rr.log(path, asset, static=True, recording=recording)
    trusted: Bool[ndarray, "t"] = confidence > trust_threshold
    if len(trusted):
        # Rows only where visibility changes; latest-at carries them.
        changes: Int64[ndarray, "k"] = np.flatnonzero(np.r_[True, trusted[1:] != trusted[:-1]])
        albedo: Float32[ndarray, "k 4"] = np.ones((len(changes), 4), dtype=np.float32)
        albedo[:, 3] = trusted[changes]
        rr.send_columns(
            path,
            indexes=[time_column(times_ns[changes]), frame_index_column(frame_indices[changes])],
            columns=rr.Asset3D.columns(albedo_factor=albedo),
            recording=recording,
        )
    rr.send_columns(
        path,
        indexes=[time_column(times_ns), frame_index_column(frame_indices)],
        columns=rr.Scalars.columns(scalars=confidence),
        recording=recording,
    )


def strip_texture_transform(glb: bytes) -> bytes:
    """Remove unsupported texture transforms, preserving all other GLB chunks.

    Rerun 0.38.1 rejects glTF files requiring KHR_texture_transform
    (checked with native Viewer pixels on 2026-09-25).

    GLB's open-ended JSON document is edited structurally; binary buffers, node
    scales and unrelated extensions are left intact.
    """
    if len(glb) < 20 or struct.unpack_from("<4sII", glb) != (b"glTF", 2, len(glb)):
        raise ValueError("invalid GLB v2 header")
    length, kind = struct.unpack_from("<I4s", glb, 12)
    if kind != b"JSON" or 20 + length > len(glb):
        raise ValueError("invalid GLB JSON chunk")
    document: dict[str, object] = json.loads(glb[20 : 20 + length])
    extension: str = "KHR_texture_transform"
    for key in ("extensionsUsed", "extensionsRequired"):
        names: object = document.get(key)
        if isinstance(names, list):
            document[key] = [name for name in names if name != extension]
            if not document[key]:
                del document[key]
    # Traverse texture infos, including those inside material extensions.
    materials: object = document.get("materials", [])
    pending: list[object] = list(materials) if isinstance(materials, list) else []
    while pending:
        value: object = pending.pop()
        if isinstance(value, dict):
            extensions: object = value.get("extensions")
            if isinstance(extensions, dict):
                extensions.pop(extension, None)
                if not extensions:
                    del value["extensions"]
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    payload: bytes = json.dumps(document, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    tail: bytes = glb[20 + length :]
    return struct.pack("<4sII", b"glTF", 2, 20 + len(payload) + len(tail)) + struct.pack("<I4s", len(payload), b"JSON") + payload + tail
