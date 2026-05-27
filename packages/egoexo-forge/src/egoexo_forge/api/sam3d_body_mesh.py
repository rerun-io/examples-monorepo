from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int32, Int64
from numpy import ndarray
from rerun.experimental import Chunk

VIDEO_TIMELINE: str = "video_time"
SAM3D_MESH_ALPHA: float = 0.35


class Sam3dBodyMeshAssetPaths(NamedTuple):
    """Local SAM-3D-Body assets needed to reconstruct MHR meshes."""

    mhr_model_path: Path
    """TorchScript MHR model path."""
    checkpoint_path: Path
    """SAM-3D-Body checkpoint path containing ``head_pose.faces``."""


class Sam3dBodyMeshReconstructor:
    """Reconstruct MHR mesh vertices from source parquet parameters."""

    def __init__(self, asset_paths: Sam3dBodyMeshAssetPaths) -> None:
        import torch

        self._torch: Any = torch
        self._mhr_model: Any = torch.jit.load(str(asset_paths.mhr_model_path), map_location="cpu")
        self._mhr_model.eval()
        self.faces: Int32[ndarray, "n_faces 3"] = _load_mesh_faces(asset_paths.checkpoint_path)

    def reconstruct_vertices(self, subject: Any) -> Float32[ndarray, "n_verts 3"]:
        """Reconstruct one subject mesh in metres."""
        if subject.shape_params is None or subject.model_params is None:
            raise ValueError("MHR mesh reconstruction requires shape_params and model_params")
        if subject.shape_params.shape != (45,):
            raise ValueError(f"Expected shape_params shape (45,), got {subject.shape_params.shape}")
        if subject.model_params.shape != (204,):
            raise ValueError(f"Expected model_params shape (204,), got {subject.model_params.shape}")

        shape_params: Any = self._torch.from_numpy(subject.shape_params[np.newaxis, :].astype(np.float32, copy=False))
        model_params: Any = self._torch.from_numpy(subject.model_params[np.newaxis, :].astype(np.float32, copy=False))
        expr_params: Any = self._torch.zeros((1, 72), dtype=self._torch.float32)
        with self._torch.no_grad():
            mhr_output: Any = self._mhr_model(shape_params, model_params, expr_params)

        vertices_tensor: Any = mhr_output[0] if isinstance(mhr_output, tuple) else mhr_output
        vertices_cm: Float32[ndarray, "1 n_verts 3"] = cast(Float32[ndarray, "1 n_verts 3"], vertices_tensor.detach().cpu().numpy())
        vertices_m: Float32[ndarray, "n_verts 3"] = np.asarray(vertices_cm[0], dtype=np.float32) / np.float32(100.0)
        return np.ascontiguousarray(vertices_m, dtype=np.float32)


def create_mesh_reconstructor(
    *,
    mhr_model_path: Path | None,
    sam3d_body_checkpoint_path: Path | None,
    require_mesh: bool,
    auto_discover_mesh_assets: bool,
) -> Sam3dBodyMeshReconstructor | None:
    asset_paths: Sam3dBodyMeshAssetPaths | None = _resolve_mesh_asset_paths(
        mhr_model_path=mhr_model_path,
        sam3d_body_checkpoint_path=sam3d_body_checkpoint_path,
        require_mesh=require_mesh,
        auto_discover_mesh_assets=auto_discover_mesh_assets,
    )
    if asset_paths is None:
        return None
    try:
        return Sam3dBodyMeshReconstructor(asset_paths)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        if require_mesh:
            raise ValueError(f"Failed to load SAM-3D-Body mesh assets from {asset_paths}") from error
        return None


def mesh_chunks(
    entity_path: str,
    *,
    records: list[tuple[float, Any]],
    mesh_reconstructor: Sam3dBodyMeshReconstructor,
    require_mesh: bool,
    timeline_name: str = VIDEO_TIMELINE,
    timeline_kind: Literal["duration", "sequence"] = "duration",
) -> list[Chunk]:
    times: list[float] = []
    vertex_positions: list[Float32[ndarray, "n_verts 3"]] = []
    vertex_normals: list[Float32[ndarray, "n_verts 3"]] = []
    triangle_indices: list[Int32[ndarray, "n_faces 3"]] = []
    albedo_factors: list[tuple[float, float, float, float]] = []
    for time_s, subject in records:
        try:
            vertices: Float32[ndarray, "n_verts 3"] = mesh_reconstructor.reconstruct_vertices(subject)
        except (RuntimeError, ValueError) as error:
            if require_mesh:
                raise ValueError(f"Failed to reconstruct MHR mesh for person_{subject.person_id}") from error
            continue
        times.append(time_s)
        vertex_positions.append(vertices)
        vertex_normals.append(_compute_vertex_normals(vertices, mesh_reconstructor.faces))
        triangle_indices.append(mesh_reconstructor.faces)
        albedo_factors.append(_mesh_albedo_factor(subject.person_id))

    if not vertex_positions:
        if require_mesh:
            raise ValueError(f"No MHR meshes could be reconstructed for {entity_path}")
        return []

    time_column: rr.TimeColumn = (
        rr.TimeColumn(timeline_name, sequence=np.asarray(times, dtype=np.int64))
        if timeline_kind == "sequence"
        else rr.TimeColumn(timeline_name, duration=np.asarray(times, dtype=np.float64))
    )
    return [
        Chunk.from_columns(
            entity_path,
            indexes=[time_column],
            columns=rr.Mesh3D.columns(
                vertex_positions=vertex_positions,
                triangle_indices=triangle_indices,
                vertex_normals=vertex_normals,
                albedo_factor=albedo_factors,
            ),
        )
    ]


def _resolve_mesh_asset_paths(
    *,
    mhr_model_path: Path | None,
    sam3d_body_checkpoint_path: Path | None,
    require_mesh: bool,
    auto_discover_mesh_assets: bool,
) -> Sam3dBodyMeshAssetPaths | None:
    if mhr_model_path is not None or sam3d_body_checkpoint_path is not None:
        resolved_model_path: Path | None = mhr_model_path
        if resolved_model_path is None and sam3d_body_checkpoint_path is not None:
            candidate_model_path: Path = sam3d_body_checkpoint_path.parent / "assets" / "mhr_model.pt"
            resolved_model_path = candidate_model_path if candidate_model_path.exists() else None

        checkpoint_path: Path | None = sam3d_body_checkpoint_path
        if checkpoint_path is None and mhr_model_path is not None:
            candidate_checkpoint_path: Path = mhr_model_path.parent.parent / "model.ckpt"
            checkpoint_path = candidate_checkpoint_path if candidate_checkpoint_path.exists() else None

        if resolved_model_path is not None and checkpoint_path is not None and resolved_model_path.exists() and checkpoint_path.exists():
            return Sam3dBodyMeshAssetPaths(resolved_model_path, checkpoint_path)
        if require_mesh:
            raise ValueError(
                "MHR mesh logging requires existing mhr_model_path and sam3d_body_checkpoint_path; "
                f"got mhr_model_path={resolved_model_path}, sam3d_body_checkpoint_path={checkpoint_path}"
            )
        return None

    if not auto_discover_mesh_assets:
        if require_mesh:
            raise ValueError("MHR mesh logging requires mesh asset paths when auto_discover_mesh_assets is false")
        return None

    for snapshot_path in _sam3d_body_hf_snapshot_candidates():
        resolved_model_path = snapshot_path / "assets" / "mhr_model.pt"
        checkpoint_path = snapshot_path / "model.ckpt"
        if resolved_model_path.exists() and checkpoint_path.exists():
            return Sam3dBodyMeshAssetPaths(resolved_model_path, checkpoint_path)

    if require_mesh:
        raise ValueError("Could not find facebook/sam-3d-body-dinov3 assets in the local Hugging Face cache")
    return None


def _sam3d_body_hf_snapshot_candidates() -> list[Path]:
    snapshot_root: Path = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--sam-3d-body-dinov3" / "snapshots"
    if not snapshot_root.exists():
        return []
    snapshots: list[Path] = [path for path in snapshot_root.iterdir() if path.is_dir()]
    return sorted(snapshots, reverse=True)


def _load_mesh_faces(checkpoint_path: Path) -> Int32[ndarray, "n_faces 3"]:
    import torch

    try:
        checkpoint: Any = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except (RuntimeError, pickle.UnpicklingError):
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict: Any = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"SAM-3D-Body checkpoint did not contain a state dict: {checkpoint_path}")

    faces_value: Any = state_dict.get("head_pose.faces")
    if faces_value is None:
        faces_value = state_dict.get("head_pose_hand.faces")
    if faces_value is None:
        raise ValueError(f"SAM-3D-Body checkpoint is missing head_pose.faces: {checkpoint_path}")

    faces_array: ndarray = cast(ndarray, faces_value.detach().cpu().numpy()) if hasattr(faces_value, "detach") else np.asarray(faces_value)
    faces: Int32[ndarray, "n_faces 3"] = np.asarray(faces_array, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected mesh faces to have shape (n_faces, 3), got {faces.shape}")
    return np.ascontiguousarray(faces, dtype=np.int32)


def _compute_vertex_normals(
    verts: Float32[ndarray, "n_verts 3"],
    faces: Int32[ndarray, "n_faces 3"],
    eps: float = 1e-12,
) -> Float32[ndarray, "n_verts 3"]:
    faces_i: Int64[ndarray, "n_faces 3"] = faces.astype(np.int64)
    v0: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 0]]
    v1: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 1]]
    v2: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 2]]
    face_normals: Float32[ndarray, "n_faces 3"] = np.cross(v1 - v0, v2 - v0)
    vertex_normals: Float32[ndarray, "n_verts 3"] = np.zeros_like(verts, dtype=np.float32)
    flat_indices: Int64[ndarray, "n_faces3"] = faces_i.reshape(-1)
    np.add.at(vertex_normals, flat_indices, np.repeat(face_normals, 3, axis=0))
    norms: Float32[ndarray, "n_verts 1"] = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    normalized: Float32[ndarray, "n_verts 3"] = (vertex_normals / np.maximum(norms, eps).astype(np.float32)).astype(np.float32)
    return np.where(norms > eps, normalized, np.float32(0.0)).astype(np.float32, copy=False)


def _mesh_albedo_factor(person_id: int) -> tuple[float, float, float, float]:
    palette: tuple[tuple[float, float, float], ...] = ((0.2, 0.6, 1.0), (1.0, 0.45, 0.2), (0.3, 0.85, 0.45), (0.85, 0.35, 1.0))
    rgb: tuple[float, float, float] = palette[person_id % len(palette)]
    return rgb[0], rgb[1], rgb[2], SAM3D_MESH_ALPHA
