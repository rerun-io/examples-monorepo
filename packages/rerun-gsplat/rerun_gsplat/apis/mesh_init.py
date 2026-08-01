"""Initialize 3DGS Gaussians from a segment's ground-truth reconstruction mesh.

ARKitScenes segments carry `world/gt/mesh` as a static, world-frame,
vertex-colored `Mesh3D` (the `_3dod_mesh.ply` reconstruction, ~2-5 cm vertex
spacing). Sampling its vertices replaces the usual COLMAP/RGB-D
backprojection initialization: one static-column fetch, no pose math.
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import torch
from arkitscenes_download.ingest.paths import GT_MESH
from jaxtyping import Float32, UInt32
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from scipy.spatial import KDTree
from torch import Tensor

from rerun_gsplat.apis.segment_views import SegmentViewsConfig


@dataclass(frozen=True, slots=True)
class ColoredPoints:
    """A colored point cloud in world frame."""

    verts_n3: Float32[ndarray, "n 3"]
    """Point positions, meters."""
    rgbs_n3: Float32[ndarray, "n 3"]
    """Colors in [0, 1]."""


@dataclass(frozen=True, slots=True)
class GaussianInit:
    """Initial Gaussian parameters, natural (non-activated) form, on CPU."""

    means_n3: Float32[Tensor, "n 3"]
    """Gaussian centers in world frame (mesh vertex positions)."""
    rgbs_n3: Float32[Tensor, "n 3"]
    """Base colors in [0, 1] (mesh vertex colors)."""
    log_scales_n3: Float32[Tensor, "n 3"]
    """Log of per-axis standard deviations, isotropic at the mean 3-NN distance."""
    quats_n4: Float32[Tensor, "n 4"]
    """Unit quaternions (wxyz, gsplat convention), identity."""
    logit_opacities_n: Float32[Tensor, "n"]
    """Inverse-sigmoid opacities."""


def unpack_rgba32(packed_n: UInt32[ndarray, "n"]) -> Float32[ndarray, "n 3"]:
    """Unpack Rerun ``Rgba32`` (0xRRGGBBAA) colors into float RGB in [0, 1]."""
    rgb_n3: Float32[ndarray, "n 3"] = np.stack(
        [(packed_n >> 24) & 0xFF, (packed_n >> 16) & 0xFF, (packed_n >> 8) & 0xFF], axis=1
    ).astype(np.float32)
    return rgb_n3 / 255.0


def load_gt_mesh(config: SegmentViewsConfig) -> ColoredPoints:
    """Fetch the segment's static gt mesh vertices and colors from the catalog.

    Static components live on no timeline, so this uses the catalog reader
    with ``index=None`` (static-only) rather than the dataloader. Only the two
    needed columns are selected — the mesh's triangle indices stay server-side.
    """
    positions_column: str = f"/{GT_MESH}:Mesh3D:vertex_positions"
    colors_column: str = f"/{GT_MESH}:Mesh3D:vertex_colors"
    dataset_entry: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset_name)
    table: pa.Table = (
        dataset_entry.filter_segments([config.segment_id])
        .filter_contents([f"/{GT_MESH}"])
        .reader(index=None)
        .select(f'"{positions_column}"', f'"{colors_column}"')
        .to_arrow_table()
    )
    if table.num_rows != 1:
        raise ValueError(f"expected one static mesh row for segment {config.segment_id!r}, got {table.num_rows}")
    positions: pa.ListScalar = table.column(positions_column)[0]
    colors: pa.ListScalar = table.column(colors_column)[0]
    verts_n3: Float32[ndarray, "n 3"] = np.asarray(positions.values.values, dtype=np.float32).reshape(-1, 3)
    packed_n: UInt32[ndarray, "n"] = np.asarray(colors.values, dtype=np.uint32)
    return ColoredPoints(verts_n3=verts_n3, rgbs_n3=unpack_rgba32(packed_n))


def gaussians_from_points(points: ColoredPoints, max_points: int, init_opacity: float = 0.5) -> GaussianInit:
    """Build initial Gaussian parameters from colored points.

    Args:
        points: Colored point cloud in world frame.
        max_points: Uniform-random subsample cap.
        init_opacity: Initial opacity (gsplat MCMC examples use 0.5).
    """
    verts_n3: Float32[ndarray, "n 3"] = points.verts_n3
    colors_n3: Float32[ndarray, "n 3"] = points.rgbs_n3
    if len(verts_n3) > max_points:
        keep: ndarray = np.random.default_rng(seed=0).choice(len(verts_n3), size=max_points, replace=False)
        verts_n3 = verts_n3[keep]
        colors_n3 = colors_n3[keep]
    # Isotropic scale at each point's mean distance to its 3 nearest neighbors
    # (the standard 3DGS init).
    distances: Float32[ndarray, "n 4"] = KDTree(verts_n3).query(verts_n3, k=4)[0].astype(np.float32)
    mean_nn_dist: Float32[ndarray, "n"] = distances[:, 1:].mean(axis=1).clip(min=1e-4)
    count: int = len(verts_n3)
    quats_n4: Float32[ndarray, "n 4"] = np.zeros((count, 4), dtype=np.float32)
    quats_n4[:, 0] = 1.0
    logit_opacity: float = float(np.log(init_opacity / (1.0 - init_opacity)))
    return GaussianInit(
        means_n3=torch.from_numpy(verts_n3),
        rgbs_n3=torch.from_numpy(colors_n3),
        log_scales_n3=torch.from_numpy(np.log(mean_nn_dist)[:, None].repeat(3, axis=1)),
        quats_n4=torch.from_numpy(quats_n4),
        logit_opacities_n=torch.full((count,), logit_opacity, dtype=torch.float32),
    )
