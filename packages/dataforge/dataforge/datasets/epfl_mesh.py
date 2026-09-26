"""Bounded EPFL FK, preserving origin-pivot translations and per-row shapes."""

from itertools import groupby
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import simplecv
from jaxtyping import Float32, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.ops.mano.mano_np import MANOLayerNP
from simplecv.ops.smplx.smplx_torch import SmplxLayerTorch

from dataforge import meshes
from dataforge.datasets.epfl_source import Fit, FitSpec, PoseRow


def corrected_translation(fits: list[Fit], root: Float32[ndarray, "3"]) -> Float32[ndarray, "n 3"]:
    """Convert shipped origin-pivot Th to the simplecv root-joint pivot."""
    rotation = Rotation.from_rotvec(np.stack([fit.parameters.Rh for fit in fits])).as_matrix()
    return (np.stack([fit.parameters.Th for fit in fits]) + np.einsum("bij,j->bi", rotation - np.eye(3), root)).astype(np.float32)


class MeshWriter:
    """Cache one model per side; rebuild on shape changes, never collapse betas."""

    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        self.mano: dict[str, MANOLayerNP] = {}
        self.body: SmplxLayerTorch | None = None
        self.betas: dict[str, Float32[ndarray, "10"]] = {}
        if not any((model_root / f"smpl/SMPL_NEUTRAL.{ext}").is_file() for ext in ("pkl", "npz")):
            raise FileNotFoundError(
                f"EPFL body_mesh requires official neutral SMPL at {model_root}/smpl/SMPL_NEUTRAL.{{pkl,npz}}; set --smpl-model-root"
            )
        for side in ("left", "right"):
            asset: Path = Path(simplecv.__file__).parent / "data" / f"MANO_{side.upper()}.pkl"
            if not asset.is_file():
                raise FileNotFoundError(f"EPFL hand_mesh requires MANO model {asset}")

    def hand_model(self, side: Literal["left", "right"], shapes: Float32[ndarray, "10"]) -> MANOLayerNP:
        """Load only shipped MANO assets; do not trigger simplecv's downloader."""
        if side not in self.mano or not np.array_equal(self.betas[side], shapes):
            self.mano[side] = MANOLayerNP(side=side, betas=shapes, use_pca=False)
            self.betas[side] = shapes.copy()
        return self.mano[side]

    def body_model(self, shapes: Float32[ndarray, "10"]) -> SmplxLayerTorch:
        """Cache neutral SMPL for the current shape coefficients."""
        if self.body is None or not np.array_equal(self.betas["body"], shapes):
            self.body = SmplxLayerTorch(betas=shapes, model_type="smpl", gender="neutral", model_root_dir=self.model_root)
            self.betas["body"] = shapes.copy()
        return self.body

    def start(self, recording: rr.RecordingStream, specs: tuple[FitSpec, ...]) -> None:
        """Log each mesh topology once, even when every row is rejected."""
        shapes: Float32[ndarray, "10"] = np.zeros(10, dtype=np.float32)
        for spec in specs:
            faces: Int64[ndarray, "f 3"] = self.body_model(shapes).faces if spec.name == "body" else self.hand_model(spec.name, shapes).f
            rr.log(spec.mesh_path, rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=spec.albedo), static=True, recording=recording)

    def write(
        self,
        recording: rr.RecordingStream,
        rows: list[PoseRow],
        times: Int64[ndarray, "n"],
        frames: Int64[ndarray, "n"],
        *,
        specs: tuple[FitSpec, ...],
    ) -> None:
        """Evaluate contiguous equal-shape runs; rejected mesh rows are empty."""
        for spec in specs:
            fits: list[Fit] = [getattr(row, spec.name) for row in rows]
            start: int = 0
            for _, run in groupby(fits, key=lambda fit: fit.parameters.shapes.tobytes()):
                group: list[Fit] = list(run)
                stop: int = start + len(group)
                valid: list[bool] = [fit.accepted for fit in group]
                accepted: list[Fit] = [fit for fit in group if fit.accepted]
                # Empty rows carry no vertices, so their unused vertex dimension is zero.
                vertices: Float32[ndarray, "n v 3"] = np.empty((0, 0, 3), dtype=np.float32)
                if accepted:
                    pose: Float32[ndarray, "n p"] = np.stack([fit.parameters.poses for fit in accepted])
                    pose[:, :3] = np.stack([fit.parameters.Rh for fit in accepted])
                    if spec.name == "body":
                        model: SmplxLayerTorch = self.body_model(group[0].parameters.shapes)
                        vertices = model.forward_batched(pose, corrected_translation(accepted, model.rest_root_joint())).vertices
                    else:
                        mano: MANOLayerNP = self.hand_model(spec.name, group[0].parameters.shapes)
                        vertices = mano(pose, corrected_translation(accepted, mano.root_trans.reshape(3)))[0]
                meshes.log_mesh_batch(
                    recording, spec.mesh_path, times_ns=times[start:stop], frame_indices=frames[start:stop], vertices=vertices, trusted=valid
                )
                start = stop
