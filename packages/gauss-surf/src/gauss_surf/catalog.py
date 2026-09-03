"""Shared Rerun catalog connection, segment reads, and layer registration."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from arkitscenes_download.ingest.cells import component_instance as _single_instance  # noqa: F401  # re-exported
from arkitscenes_download.ingest.paths import TIMELINE
from arkitscenes_download.ingest.timestamps import TimedeltaNs, match_exact_timestamps, table_timestamps  # noqa: F401  # re-exported
from beartype.roar import BeartypeException
from datafusion import col
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView, OnDuplicateSegmentLayer, RegistrationHandle
from simplecv.rerun_dataloader import SegmentNvdecDecoder
from torch import Tensor

DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:51236"
"""Development catalog server URL for gauss-surf."""
DEFAULT_DATASET_NAME: str = "arkitscenes-v2"
"""Development ARKitScenes dataset used by the staged gauss-surf pipeline."""

def _matrix_from_cell(value: Any, component_name: str) -> Float32[ndarray, "3 3"]:
    """Decode one column-major Rerun 3x3 matrix cell."""
    return np.asarray(_single_instance(value, component_name), dtype=np.float32).reshape(3, 3, order="F")


def _resolution_from_cell(value: Any, component_name: str) -> tuple[int, int]:
    """Decode one Rerun resolution cell as width and height."""
    resolution_2: Float32[ndarray, "2"] = np.asarray(_single_instance(value, component_name), dtype=np.float32)
    return int(round(float(resolution_2[0]))), int(round(float(resolution_2[1])))


def connect_catalog(
    catalog_url: str = DEFAULT_CATALOG_URL,
    dataset_name: str = DEFAULT_DATASET_NAME,
    *,
    create_missing: bool = False,
) -> CatalogClient:
    """Connect to a catalog and optionally create the requested dataset.

    Args:
        catalog_url: Rerun catalog server URL.
        dataset_name: Dataset to resolve on the server.
        create_missing: Create ``dataset_name`` when it does not exist.

    Returns:
        Connected catalog client.
    """
    try:
        client: CatalogClient = CatalogClient(catalog_url)
        dataset_names: list[str] = client.dataset_names()
        if create_missing and dataset_name not in dataset_names:
            client.create_dataset(dataset_name, exist_ok=True)
    except BeartypeException:
        raise
    except Exception as error:
        raise SystemExit(f"catalog not reachable at {catalog_url} — start it with `pixi run gauss-surf-serve`") from error
    if not create_missing and dataset_name not in dataset_names:
        raise SystemExit(f"dataset {dataset_name!r} is absent — seed it with `pixi run gauss-surf-register-segment --video-id <id>`")
    return client


@dataclass(slots=True)
class SegmentReader:
    """Composable reads for one existing catalog segment."""

    dataset: DatasetEntry
    """Catalog dataset entry containing the bound segment."""
    video_id: str
    """Exact catalog segment identifier."""
    _cached_row: dict[str, Any] | None = field(default=None, init=False, repr=False)
    """Lazily cached segment-table row."""

    @classmethod
    def open(
        cls,
        catalog_url: str,
        dataset_name: str,
        video_id: str,
    ) -> "SegmentReader":
        """Connect to an existing dataset and bind a reader to one segment.

        Args:
            catalog_url: Rerun catalog server URL.
            dataset_name: Existing dataset name.
            video_id: Exact segment identifier.

        Returns:
            A reader whose individual methods load only the requested data.
        """
        client: CatalogClient = connect_catalog(catalog_url, dataset_name)
        dataset: DatasetEntry = client.get_dataset(dataset_name)
        return cls(dataset, video_id)

    def row(self) -> dict[str, Any]:
        """Return the unique segment-table row, loading and caching it on first use."""
        if self._cached_row is not None:
            return self._cached_row
        segment_table: pa.Table = pa.Table.from_batches(self.dataset.segment_table().collect())
        rows: list[dict[str, Any]] = segment_table.to_pylist()
        matching_rows: list[dict[str, Any]] = [row for row in rows if str(row["rerun_segment_id"]) == self.video_id]
        if not matching_rows:
            available_ids: list[str] = [str(row["rerun_segment_id"]) for row in rows]
            raise SystemExit(f"video id {self.video_id!r} is absent; available ids: {', '.join(available_ids)}")
        if len(matching_rows) != 1:
            raise SystemExit(f"catalog dataset has {len(matching_rows)} rows for segment {self.video_id!r}")
        self._cached_row = matching_rows[0]
        return self._cached_row

    def require_layers(self, required_layers: tuple[str, ...]) -> None:
        """Exit cleanly when the segment lacks any requested upstream layer.

        Args:
            required_layers: Layer names the caller needs before it can run.
        """
        layer_names: set[str] = {str(name) for name in self.row()["rerun_layer_names"]}
        missing_layers: list[str] = sorted(set(required_layers) - layer_names)
        if missing_layers:
            raise SystemExit(f"segment {self.video_id} is missing required layers: {missing_layers}")

    def require_zero_orientation(self, stage_context: str) -> None:
        """Require the baked zero-quarter-turn camera orientation.

        Args:
            stage_context: Human-readable stage constraint used in errors.
        """
        property_name: str = "property:capture:orientation_quarter_turns_ccw"
        property_value: Any = self.row().get(property_name)
        if isinstance(property_value, (list, np.ndarray)):
            if len(property_value) == 0:
                raise SystemExit(f"segment {self.video_id} has an empty catalog property {property_name!r}")
            property_value = property_value[0]
        if property_value is None:
            raise SystemExit(f"segment {self.video_id} is missing required catalog property {property_name!r}")
        quarter_turns: int = int(property_value)
        if quarter_turns != 0:
            raise SystemExit(
                f"segment {self.video_id} has {property_name}={quarter_turns}; {stage_context} supports only zero quarter-turns"
            )

    def segment_view(self) -> DatasetView:
        """Return the catalog query view filtered to this segment."""
        self.row()
        return self.dataset.filter_segments(self.video_id)

    def chosen_table(self, chosen_column: str, columns: tuple[str, ...]) -> pa.Table:
        """Read chosen timestamps and requested latest-at columns in time order.

        Args:
            chosen_column: Sparse component whose non-null rows define selection.
            columns: Components to join at each chosen timestamp.

        Returns:
            Non-empty Arrow table containing ``time`` and ``columns``.
        """
        segment_view: DatasetView = self.segment_view()
        available_columns: set[str] = set(segment_view.arrow_schema().names)
        missing_columns: list[str] = [name for name in (chosen_column, *columns) if name not in available_columns]
        if missing_columns:
            raise SystemExit(f"segment {self.video_id} is missing required chosen-frame columns: {missing_columns}")
        table: pa.Table = (
            segment_view.reader(index=TIMELINE, fill_latest_at=True)
            .filter(col(f'"{chosen_column}"').is_not_null())
            .select(TIMELINE, *columns)
            .sort(TIMELINE)
            .to_arrow_table()
        )
        if table.num_rows == 0:
            raise SystemExit(f"segment {self.video_id} selection {chosen_column!r} contains no rows")
        return table

    def pose_table(self, translation_column: str, quaternion_column: str) -> pa.Table:
        """Read a non-empty pose track in timestamp order.

        Args:
            translation_column: Transform translation component.
            quaternion_column: Transform quaternion component.

        Returns:
            Arrow table containing exact pose timestamps and both components.
        """
        segment_view: DatasetView = self.segment_view()
        available_columns: set[str] = set(segment_view.arrow_schema().names)
        missing_columns: list[str] = [name for name in (translation_column, quaternion_column) if name not in available_columns]
        if missing_columns:
            raise SystemExit(f"segment {self.video_id} is missing required pose columns: {missing_columns}")
        table: pa.Table = (
            segment_view.reader(index=TIMELINE, fill_latest_at=False)
            .filter(col(f'"{translation_column}"').is_not_null() & col(f'"{quaternion_column}"').is_not_null())
            .select(TIMELINE, translation_column, quaternion_column)
            .sort(TIMELINE)
            .to_arrow_table()
        )
        if table.num_rows == 0:
            raise SystemExit(f"segment {self.video_id} has no pose track")
        return table

    def decode_frames(
        self,
        entity_path: str,
        timestamps_n: TimedeltaNs,
        *,
        fps: float,
        device: torch.device,
    ) -> Iterator[UInt8[Tensor, "3 h w"]]:
        """Decode video frames at exact requested timestamps.

        Args:
            entity_path: VideoStream entity to decode.
            timestamps_n: Requested ``timedelta64[ns]`` timestamps shaped ``n``.
            fps: Nominal packet rate used by the decoder.
            device: Torch device that receives decoded frames.

        Yields:
            uint8 RGB frames shaped ``3 h w`` in requested timestamp order.
        """
        decoder: SegmentNvdecDecoder = SegmentNvdecDecoder(self.dataset, entity_path, TIMELINE, device, int(fps))
        timestamps_verified: bool = False
        for timestamp in timestamps_n:
            frame_chw: UInt8[Tensor, "3 h w"] | None = decoder.decode_at(timestamp, self.video_id)
            if frame_chw is None:
                raise RuntimeError(f"{entity_path} decoder returned no frame at requested timestamp {timestamp}")
            if not timestamps_verified:
                matched_indices_n: Int64[ndarray, "n"] = match_exact_timestamps(decoder.times, timestamps_n)
                if len(np.unique(matched_indices_n)) != len(matched_indices_n):
                    raise RuntimeError(f"requested {entity_path} timestamps do not map one-to-one onto video packets")
                timestamps_verified = True
            yield frame_chw


def register_layer(dataset: DatasetEntry, rrd_path: Path, layer_name: str) -> None:
    """Register one layer RRD with clean replacement semantics.

    Args:
        dataset: Target catalog dataset.
        rrd_path: Complete local layer recording.
        layer_name: Catalog layer name assigned to the recording.
    """
    if not rrd_path.is_file():
        raise FileNotFoundError(rrd_path)
    registration: RegistrationHandle = dataset.register(
        [rrd_path.resolve().as_uri()],
        layer_name=layer_name,
        on_duplicate=OnDuplicateSegmentLayer.REPLACE,
    )
    registration.wait()
