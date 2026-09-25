"""Textured HO-Cap OBJ adapter, without extraction or an extra dependency."""

import json
import struct
from zipfile import ZipFile

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray


def textured_glb(archive: ZipFile, object_id: str) -> bytes:
    """Build an in-memory GLB with one vertex per unique OBJ face corner.

    The release contains triangular Open3D exports with positive v/vt/vn indices
    and one PNG material. Reject other layouts rather than silently lose texture.
    OBJ v increases upwards; Rerun texture coordinates increase downwards.
    """
    prefix: str = f"models/{object_id}"
    material: str = archive.read(f"{prefix}/textured_mesh.mtl").decode()
    textures: list[str] = [line.split(maxsplit=1)[1] for line in material.splitlines() if line.startswith("map_Kd ")]
    if textures != ["textured_mesh_0.png"]:
        raise ValueError(f"{prefix}: expected one textured_mesh_0.png material")
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    uv: list[list[float]] = []
    corners: list[tuple[int, int, int]] = []
    for line in archive.read(f"{prefix}/textured_mesh.obj").decode().splitlines():
        fields: list[str] = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "vn":
            normals.append([float(value) for value in fields[1:4]])
        elif fields[0] == "vt":
            uv.append([float(fields[1]), 1.0 - float(fields[2])])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError(f"{prefix}: non-triangular OBJ face")
            for field in fields[1:]:
                indices: list[int] = [int(value) - 1 for value in field.split("/")]
                if len(indices) != 3 or min(indices) < 0:
                    raise ValueError(f"{prefix}: expected positive v/vt/vn face indices")
                corners.append((indices[0], indices[1], indices[2]))
    if not corners:
        raise ValueError(f"{prefix}: empty mesh")
    corner_indices: Int64[ndarray, "c 3"] = np.asarray(corners, dtype=np.int64)
    unique, inverse = np.unique(corner_indices, axis=0, return_inverse=True)
    positions: Float32[ndarray, "v 3"] = np.asarray(vertices, dtype=np.float32)[unique[:, 0]]
    directions: Float32[ndarray, "v 3"] = np.asarray(normals, dtype=np.float32)[unique[:, 2]]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    texcoords: Float32[ndarray, "v 2"] = np.asarray(uv, dtype=np.float32)[unique[:, 1]]
    binary = bytearray()
    views: list[dict[str, int]] = []
    for payload, target in (
        (positions.astype("<f4").tobytes(), 34962),
        (directions.astype("<f4").tobytes(), 34962),
        (texcoords.astype("<f4").tobytes(), 34962),
        (inverse.astype("<u4").tobytes(), 34963),
        (archive.read(f"{prefix}/{textures[0]}"), None),
    ):
        binary.extend(b"\0" * (-len(binary) % 4))
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        views.append(view)
        binary.extend(payload)
    document: dict[str, object] = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": 5126, "count": len(directions), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(texcoords), "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": len(inverse), "type": "SCALAR"},
        ],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "samplers": [{}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    encoded: bytes = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    return (
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + bytes(binary)
    )
