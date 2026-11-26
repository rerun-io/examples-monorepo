"""Temporal rows held as numpy arrays per entity and sent with ``send_columns`` in blocks, instead of one ``rr.log`` each.

A run's layer has some forty entities that gain a row every frameset. One ``rr.log`` per row builds an archetype, converts
each component to Arrow and pushes it through the SDK's batcher; over a 54 k-frameset RoboCap window that cost the
replay half its speed. :class:`ColumnLog` keeps each entity's rows as arrays, and :meth:`ColumnLog.flush` sends each
entity's held rows as one chunk: one Arrow conversion per component per block. The caller flushes every few hundred
framesets, so the memory stays bounded on a four-hour run.

Rows carry their own stamp on ``video_time``: nothing reads the global time cursor.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import rerun as rr
from jaxtyping import Int64
from numpy import ndarray

from slam_rs.rig import TIMELINE

RowMode: TypeAlias = Literal["one", "many"]
"""``one``: each row holds one instance of every field (a pose, a scalar). ``many``: each row holds a batch, the same
count in every field (a point cloud), sent partitioned by those counts."""


ColumnsOf: TypeAlias = Callable[..., rr.ComponentColumnList]
"""An archetype's ``columns`` classmethod (``rr.Points3D.columns``): what turns the held fields into component columns."""


@dataclass(slots=True)
class EntityRows:
    """One entity's held rows of one archetype."""

    columns_of: ColumnsOf
    """The archetype's ``columns``, which the rows go through."""
    mode: RowMode
    """How many instances a row holds (:data:`RowMode`)."""
    t_ns: list[int] = field(default_factory=list)
    """Each row's ``video_time``, nanoseconds."""
    counts: list[int] = field(default_factory=list)
    """Instances in each row (``many`` only)."""
    fields: dict[str, list[Any]] = field(default_factory=dict)
    """Per field: the rows' values (``one``), or their instances in row order (``many``)."""


@dataclass(slots=True)
class ColumnLog:
    """Rows per entity, sent as columns at :meth:`flush`, onto ``recording`` (None: the global recording)."""

    recording: rr.RecordingStream | None = None
    """Where :meth:`flush` sends; None is the global recording, which a tool's ``RerunTyroConfig`` set up."""
    rows: dict[tuple[str, str, RowMode], EntityRows] = field(default_factory=dict)
    """Held rows by entity path, archetype and mode, in the order they first came."""
    styled: set[str] = field(default_factory=set)
    """Entities whose static style (:meth:`static_once`) is already logged."""

    def static_once(self, entity: str, style: rr.AsComponents) -> None:
        """Log what every row of ``entity`` shares (a radius, a colour) once, static, the first time it is asked for.

        Args:
            entity: Entity path.
            style: The shared components, e.g. ``rr.Points3D.from_fields(radii=0.01)``.
        """
        if entity not in self.styled:
            self.styled.add(entity)
            rr.log(entity, style, static=True, recording=self.recording)

    def add(self, entity: str, columns_of: ColumnsOf, t_ns: int, **fields: Any) -> None:
        """Hold one row with one instance of each field: a ``Transform3D``, a ``Scalars`` value, a ``Clear``.

        Args:
            entity: Entity path.
            columns_of: The archetype's ``columns`` (``rr.Transform3D.columns``), which takes these fields.
            t_ns: The row's ``video_time``, nanoseconds.
            **fields: One value per field (a translation, a scalar); every row of the entity names the same fields.
        """
        held: EntityRows = self._held(entity, columns_of, "one")
        held.t_ns.append(t_ns)
        for name, value in fields.items():
            held.fields.setdefault(name, []).append(value)

    def add_batch(self, entity: str, columns_of: ColumnsOf, t_ns: int, **fields: Sequence[Any] | ndarray) -> None:
        """Hold one row of many instances: a point cloud, a set of strips.

        Args:
            entity: Entity path.
            columns_of: The archetype's ``columns`` (``rr.Transform3D.columns``), which takes these fields.
            t_ns: The row's ``video_time``, nanoseconds.
            **fields: The row's instances per field (arrays with the instance axis first, or a list of strips), the same
                count in each.
        """
        held: EntityRows = self._held(entity, columns_of, "many")
        counts: set[int] = {len(value) for value in fields.values()}
        if len(counts) != 1:
            raise ValueError(f"{entity}: a row's fields hold different instance counts {sorted(counts)}")
        held.t_ns.append(t_ns)
        held.counts.append(counts.pop())
        for name, value in fields.items():
            held.fields.setdefault(name, []).append(value)

    def flush(self) -> None:
        """Send every held row, one ``send_columns`` per entity and archetype, and forget them."""
        for (entity, _, mode), held in self.rows.items():
            if not held.t_ns:
                continue
            times: Int64[ndarray, " n"] = np.asarray(held.t_ns, dtype=np.int64)
            columns: rr.ComponentColumnList
            if mode == "one":
                columns = held.columns_of(**{name: _stacked(values) for name, values in held.fields.items()})
            else:
                columns = held.columns_of(**{name: _joined(values) for name, values in held.fields.items()}).partition(held.counts)
            rr.send_columns(entity, indexes=[rr.TimeColumn(TIMELINE, duration=times.astype("timedelta64[ns]"))], columns=columns, recording=self.recording)
        self.rows.clear()

    def _held(self, entity: str, columns_of: ColumnsOf, mode: RowMode) -> EntityRows:
        key: tuple[str, str, RowMode] = (entity, columns_of.__qualname__, mode)
        held: EntityRows | None = self.rows.get(key)
        if held is None:
            held = self.rows[key] = EntityRows(columns_of=columns_of, mode=mode)
        return held


def _stacked(values: list[Any]) -> Any:
    """One value per row as a column: arrays stacked, anything else (strings, bools, ``None``-free scalars) as a list."""
    return np.stack(values) if isinstance(values[0], ndarray) else values


def _joined(values: list[Any]) -> Any:
    """Every row's instances end to end: arrays concatenated, lists (strips of different lengths) chained."""
    if isinstance(values[0], ndarray):
        return np.concatenate(values)
    return [instance for value in values for instance in value]
