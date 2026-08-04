"""DEPRECATED v1 reader for flat ``/world/{exo,ego}/{name}`` exoego ``.rrd`` files.

The exoego writer now emits the COLMAP-style **``exoego:v2``** rig layout
(``/world/rig_NN/cam_MM``; see :mod:`simplecv.rerun_rig_logger` and
``packages/simplecv/docs/exoego_schema.md``). This reader still parses the old
flat layout, so it correctly reads **pre-existing v1 ``.rrd``** but **cannot read
v2 writer output**. A v2 reader is a pending follow-up (a full reader refactor);
until it lands, do not re-ingest freshly generated ``.rrd`` through this path.
``RRDSequence`` emits a :class:`DeprecationWarning` on construction.
"""

import warnings
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.experimental as rre
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from rerun.catalog import Schema
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.rrd_ego import RRDEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.rrd_exo import RRDExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, EnvironmentMesh, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.rrd_query_utils import RRDQuerySession, series_to_int64_ns


@dataclass
class RRDExoEgoConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: RRDSequence)
    rrd_path: Path = Path("/path/to/rrd/file.rrd")
    load_labels: bool = True
    # Required: .rrd file produced by tools/t265_slam.py

    def setup(self, **kwargs: object) -> None:
        """Reject obsolete RRD ingestion before opening the recording."""
        del kwargs
        raise RuntimeError(
            "RRD dataset setup is disabled: re-ingesting video from legacy RRD recordings is obsolete. "
            "Use the original filesystem MP4 dataset instead."
        )


class RRDSequence(BaseExoEgoSequence[RRDExoEgoConfig]):
    _recording: rre.LazyStore | None = None

    def __init__(self, cfg: RRDExoEgoConfig) -> None:
        warnings.warn(
            "RRDSequence reads the DEPRECATED flat exoego v1 layout (/world/{exo,ego}/{name}). "
            "The writer now emits the exoego:v2 rig layout (/world/rig_NN/cam_MM); this reader "
            "cannot parse v2 output. A v2 reader refactor is pending — see rrd_exoego module docs.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Load once and share with ego/exo/labels.
        self._recording = rre.RrdReader(cfg.rrd_path).store()
        self._query_session = RRDQuerySession(cfg.rrd_path)
        super().__init__(cfg)

    def __getitem__(
        self,
        idx: int | None = None,
        ts_nano: np.timedelta64 | None = None,
    ) -> ExoEgoSample:
        """
        Fetch a time-synchronised ego/exo sample aligned to the canonical timeline.
        """
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        exo_cam_params_list, exo_bgr_list = self._sample_exo(ts_ns)
        ego_depth_list = self._sample_ego_depths(ts_ns)
        exo_depth_list = self._sample_exo_depths(ts_ns)
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            ego_depth_list=ego_depth_list,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            exo_depth_list=exo_depth_list,
            labels=labels,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.canonical_timestamps_ns.shape[0])

    def _build_ego(self) -> BaseEgoSequence[RRDExoEgoConfig] | None:
        try:
            ego_seq = RRDEgoSequence(
                self.config,
                recording=self._recording,
                query_session=self._query_session,
            )
            return ego_seq
        except AssertionError as exc:
            if "No ego camera streams" in str(exc):
                return None
            raise

    def _build_exo(self) -> BaseExoSequence[RRDExoEgoConfig] | None:
        try:
            exo_seq = RRDExoSequence(
                self.config,
                recording=self._recording,
                query_session=self._query_session,
            )
            return exo_seq
        except AssertionError as exc:
            if "No exo camera streams" in str(exc):
                return None
            raise

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego/exo videos (and labels if present)."""

        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()
        self._exo_stream_names.clear()

        if self.ego_sequence is not None:
            # Get video blobs if available (RRD sequences), otherwise use paths
            ego_blobs: dict[str, bytes] | None = getattr(self.ego_sequence, "_video_blobs", None)
            for name in self.ego_sequence.ego_video_names:
                stream_name: str = f"ego/{name}"
                if ego_blobs and name in ego_blobs:
                    # Use blob directly for RRD sequences
                    timestamps: Int[ndarray, "n_frames"] = rr.AssetVideo(
                        contents=ego_blobs[name]
                    ).read_frame_timestamps_nanos()
                else:
                    # Fall back to path for non-RRD sequences
                    idx: int = self.ego_sequence.ego_video_names.index(name)
                    video_path = self.ego_sequence.ego_video_paths[idx]
                    timestamps = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._ego_stream_names.append(stream_name)

        if self.exo_sequence is not None:
            # Get video blobs if available (RRD sequences), otherwise use paths
            exo_blobs: dict[str, bytes] | None = getattr(self.exo_sequence, "_video_blobs", None)
            for name in self.exo_sequence.exo_video_names:
                stream_name = f"exo/{name}"
                if exo_blobs and name in exo_blobs:
                    # Use blob directly for RRD sequences
                    timestamps = rr.AssetVideo(contents=exo_blobs[name]).read_frame_timestamps_nanos()
                else:
                    # Fall back to path for non-RRD sequences
                    idx = self.exo_sequence.exo_video_names.index(name)
                    video_path = self.exo_sequence.exo_video_paths[idx]
                    timestamps = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._exo_stream_names.append(stream_name)

        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is not None and labels.timestamps_ns is not None:
            stream_ts["labels"] = labels.timestamps_ns

        return stream_ts

    def load_labels(self) -> ExoEgoLabels | None:
        """Load COCO-133 3D keypoints and confidences from the RRD recording."""
        rrd_path: Path = self.config.rrd_path
        assert rrd_path.exists(), f"RRD path {rrd_path} does not exist"

        if self._recording is None:
            self._recording = rre.RrdReader(rrd_path).store()

        timeline: str = "video_time"
        entity_path: str = "world/gt/coco133_xyz"
        df: pd.DataFrame = self._query_session.read_pandas(
            contents=entity_path,
            selectors=[
                f"{entity_path}:Points3D:positions",
                f"{entity_path}:simplecv.KeypointConfidence3D:confidences",
            ],
            index=timeline,
        )

        positions_series: pd.DataFrame | pd.Series | None = df[f"/{entity_path}:Points3D:positions"]
        confidences_series: pd.DataFrame | pd.Series | None = df[
            f"/{entity_path}:simplecv.KeypointConfidence3D:confidences"
        ]
        timestamps_series: pd.Series | None = df.get(timeline)

        if positions_series is None or confidences_series is None:
            return None

        positions_arrays: list[Float32[ndarray, "133 3"]] = [
            np.stack(entry, axis=0).astype(np.float32, copy=False) for entry in positions_series.to_numpy()
        ]
        xyz_stack: Float32[ndarray, "num_frames 133 3"] = np.stack(positions_arrays, axis=0)

        confidences_arrays: list[Float32[ndarray, "133"]] = [
            np.asarray(entry, dtype=np.float32) for entry in confidences_series.to_numpy()
        ]
        conf_stack: Float32[ndarray, "num_frames 133"] = np.stack(confidences_arrays, axis=0)

        timestamps_ns: Int[ndarray, "num_frames"] | None = None
        if timestamps_series is not None:
            timestamps_ns = np.asarray(series_to_int64_ns(timestamps_series), dtype=np.int64)

        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.concatenate(
            [xyz_stack, conf_stack[..., np.newaxis]],
            axis=-1,
        )
        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
            timestamps_ns=timestamps_ns,
        )

    def load_environment_mesh(self) -> EnvironmentMesh | None:
        """Load the static environment mesh from the recording, if any."""
        rrd_path: Path = self.config.rrd_path
        if not rrd_path.exists():
            return None

        recording: rre.LazyStore | None = self._recording
        assert recording is not None, f"RRD recording at {rrd_path} could not be loaded."
        schema: Schema = recording.schema()
        entity_path: str = "world/gt/env_mesh"

        available_components: set[str] = self._available_mesh_components(schema, entity_path)
        if (
            "Mesh3D:vertex_positions" not in available_components
            or "Mesh3D:triangle_indices" not in available_components
        ):
            return None

        selectors: list[str] = [
            f"{entity_path}:Mesh3D:vertex_positions",
            f"{entity_path}:Mesh3D:triangle_indices",
        ]
        include_normals: bool = "Mesh3D:vertex_normals" in available_components
        include_colors: bool = "Mesh3D:vertex_colors" in available_components
        if include_normals:
            selectors.append(f"{entity_path}:Mesh3D:vertex_normals")
        if include_colors:
            selectors.append(f"{entity_path}:Mesh3D:vertex_colors")

        candidate_timelines: list[str | None] = []
        if self.exo_sequence is not None:
            candidate_timelines.append(getattr(self.exo_sequence, "_video_timeline", None))
        candidate_timelines.extend(["video_time", "log_time", "log_tick"])

        examined: set[str | None] = set()
        for timeline in candidate_timelines:
            if timeline is None or timeline in examined:
                continue
            examined.add(timeline)
            samples: list[dict[str, object]] = self._read_mesh_samples_from_view(
                timeline=timeline,
                selectors=selectors,
                query_session=self._query_session,
                entity_path=entity_path,
            )

            for sample in samples:
                positions = self._parse_vertex_positions(sample.get(f"{entity_path}:Mesh3D:vertex_positions"))
                triangles = self._parse_triangle_indices(sample.get(f"{entity_path}:Mesh3D:triangle_indices"))
                if positions is None or triangles is None:
                    continue

                normals = self._parse_vertex_normals(
                    sample.get(f"{entity_path}:Mesh3D:vertex_normals"),
                    expected_vertices=len(positions),
                )
                colors = self._parse_vertex_colors(
                    sample.get(f"{entity_path}:Mesh3D:vertex_colors"),
                    expected_vertices=len(positions),
                )

                return EnvironmentMesh(
                    vertex_positions=positions,
                    triangle_indices=triangles,
                    vertex_normals=normals,
                    vertex_colors=colors,
                )
        return None

    @staticmethod
    def _available_mesh_components(schema: Schema, entity_path: str) -> set[str]:
        components: set[str] = set()
        for descriptor in schema.component_columns():
            entity = getattr(descriptor, "entity_path", None)
            component = getattr(descriptor, "component", None)
            if entity is None or component is None:
                continue
            entity_str = str(entity).lstrip("/")
            if entity_str == entity_path:
                components.add(str(component))
        return components

    @staticmethod
    def _parse_vertex_positions(entry: object) -> Float32[ndarray, "num_vertices 3"] | None:
        if entry is None:
            return None
        positions = np.asarray(entry, dtype=np.float32)
        positions = np.squeeze(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            return None
        return np.ascontiguousarray(positions.astype(np.float32), dtype=np.float32)

    @staticmethod
    def _parse_triangle_indices(entry: object) -> Int[ndarray, "num_faces 3"] | None:
        if entry is None:
            return None
        triangles = np.asarray(entry, dtype=np.int32)
        triangles = np.squeeze(triangles)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            return None
        return np.ascontiguousarray(triangles.astype(np.int32), dtype=np.int32)

    @staticmethod
    def _parse_vertex_normals(
        entry: object,
        *,
        expected_vertices: int,
    ) -> Float32[ndarray, "num_vertices 3"] | None:
        if entry is None:
            return None
        normals = np.asarray(entry, dtype=np.float32)
        normals = np.squeeze(normals)
        if normals.ndim != 2 or normals.shape[1] != 3:
            return None
        normals = normals[:expected_vertices]
        return np.ascontiguousarray(normals.astype(np.float32), dtype=np.float32)

    @staticmethod
    def _parse_vertex_colors(
        entry: object,
        *,
        expected_vertices: int,
    ) -> UInt8[ndarray, "num_vertices 4"] | None:
        if entry is None:
            return None
        colors_np = np.asarray(entry)
        if colors_np.size == 0:
            return None

        colors_np = np.squeeze(colors_np)

        if colors_np.ndim == 1:
            colors_uint32 = colors_np.astype(np.uint32, copy=False)
            colors = np.empty((colors_uint32.shape[0], 4), dtype=np.uint8)
            colors[:, 0] = (colors_uint32 >> 24) & 0xFF
            colors[:, 1] = (colors_uint32 >> 16) & 0xFF
            colors[:, 2] = (colors_uint32 >> 8) & 0xFF
            colors[:, 3] = colors_uint32 & 0xFF
        elif colors_np.ndim == 2 and colors_np.shape[1] in (3, 4):
            if np.issubdtype(colors_np.dtype, np.floating):
                try:
                    max_value = float(np.nanmax(colors_np))
                except ValueError:
                    max_value = 1.0
                if max_value <= 1.0:
                    colors_np = np.nan_to_num(colors_np, nan=0.0)
                    colors_np = np.clip(colors_np, 0.0, 1.0) * 255.0
            colors_np = colors_np.astype(np.uint8, copy=False)
            if colors_np.shape[1] == 3:
                alpha = np.full((colors_np.shape[0], 1), 255, dtype=np.uint8)
                colors = np.concatenate([colors_np, alpha], axis=1)
            else:
                colors = colors_np
        else:
            return None

        if colors.shape[0] > expected_vertices:
            colors = colors[:expected_vertices]
        return np.ascontiguousarray(colors.astype(np.uint8), dtype=np.uint8)

    @staticmethod
    def _read_mesh_samples_from_view(
        *,
        timeline: str,
        selectors: list[str],
        query_session: RRDQuerySession,
        entity_path: str,
    ) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []

        table_static = query_session.read_arrow(
            contents=entity_path,
            selectors=selectors,
            index=None,
        )
        if table_static.num_rows > 0:
            column_data = {
                selector: table_static.column(idx).combine_chunks().to_pylist()
                for idx, selector in enumerate(selectors)
            }
            for row_idx in range(table_static.num_rows):
                samples.append({selector: column_data[selector][row_idx] for selector in selectors})
            return samples

        table_dynamic = query_session.read_arrow(
            contents=entity_path,
            selectors=selectors,
            index=timeline,
        )
        if table_dynamic.num_rows == 0:
            return samples

        column_data = {
            selector: column.combine_chunks().to_pylist()
            for selector, column in zip(selectors, table_dynamic.columns[1:], strict=True)
        }
        for row_idx in range(table_dynamic.num_rows):
            samples.append({selector: column_data[selector][row_idx] for selector in selectors})
        return samples

    @classmethod
    def iter_episode_sequences(cls, cfg: RRDExoEgoConfig) -> Generator["RRDSequence", None, None]:
        raise NotImplementedError("RRDSequence.iter_episode_sequences is not implemented.")

    @classmethod
    def num_sequences_for_config(cls, cfg: RRDExoEgoConfig) -> int:
        raise NotImplementedError("RRDSequence.num_sequences_for_config is not implemented.")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RUF

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        if self.exo_sequence is not None:
            return self.exo_sequence.image_plane_distance
        return 0.1
