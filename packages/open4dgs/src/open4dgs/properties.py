"""The recording properties open4dgs writes and reads, one ``@serde`` record each: the ``moge2_prior``, ``people_matte``
and ``init_roma`` layers and the init cloud here, the run (``open4dgs-train``) in ``run_log.RunProperties``.

Each record's ``to_properties()`` is what goes into ``rr.AnyValues`` (key names and value encodings as every recording
already on the catalog has them), and ``from_cells`` reads it back from property cells: one list of values per field,
as ``rrd_cells`` (a recording) and ``row_cells`` (a catalog segment-table row) return them. Writers and readers use
the same class, so the encoding exists in one place. Records read from older recordings get the defaults of the
fields those recordings lack.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import rerun as rr
from rerun.chunk import RrdReader
from serde import from_dict, serde

Cells: TypeAlias = Mapping[str, Sequence[object]]
"""A property's fields: one list of values per field (a scalar field holds one value)."""
PropertyValue: TypeAlias = str | int | float | bool | list[str] | list[float]


def any_values(properties: Mapping[str, PropertyValue]) -> rr.AnyValues:
    """A record's ``to_properties()`` as the ``rr.AnyValues`` that ``send_property`` takes."""
    return rr.AnyValues(**properties)  # pyrefly: ignore[bad-argument-type]  # the stub types **kwargs against drop_untyped_nones


def rrd_cells(reader: RrdReader, name: str) -> dict[str, list[object]]:
    """The cells of property ``name`` in a recording (empty when it has none)."""
    out: dict[str, list[object]] = {}
    for chunk in reader.stream().filter(content=f"/__properties/{name}").to_chunks():
        batch = chunk.to_record_batch()
        for column in batch.schema.names:
            values = batch.column(column).to_pylist()[0]
            if isinstance(values, list):
                out[column.split(":")[-1]] = values
    return out


def row_cells(row: Mapping[str, object], name: str) -> dict[str, list[object]]:
    """The cells of property ``name`` in a catalog segment-table row (``property:<name>:<field>`` columns; a cell comes
    back as ``[[values]]``, ``[values]`` or None)."""
    prefix: str = f"property:{name}:"
    out: dict[str, list[object]] = {}
    for column, cell in row.items():
        if not column.startswith(prefix) or cell is None:
            continue
        values: object = cell
        while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        out[column.removeprefix(prefix)] = list(values) if isinstance(values, list) else [values]
    return out


def one(cells: Cells, key: str, default: object = None) -> object:
    """The value of scalar field ``key`` (``default`` when the property lacks it)."""
    values: Sequence[object] | None = cells.get(key)
    return values[0] if values else default


def floats(text: str) -> tuple[float, ...]:
    """A comma-joined float list (the encoding of tuple fields in properties)."""
    return tuple(float(v) for v in text.split(",")) if text else ()


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RoiSignature:
    """The grid of one ROI camera, at the precision the prior layer stores it (K to 0.01 px, R to 1e-4)."""

    pad: float
    max_side: int | None
    width: int
    height: int
    K: tuple[float, float, float, float]
    """fx, fy, cx, cy."""
    R: tuple[float, ...]
    """``cam_R_virtual``, row-major 3 x 3."""

    @classmethod
    def of(cls, pad: float, max_side: int | None, width: int, height: int, K: Sequence[float], R: Sequence[float]) -> "RoiSignature":
        return cls(pad, max_side, width, height, tuple(round(float(v), 2) for v in K), tuple(round(float(v), 4) for v in R))  # type: ignore[arg-type]

    def encode(self) -> str:
        k: str = ",".join(f"{v:.2f}" for v in self.K)
        return f"pad={self.pad:g};max_side={self.max_side};{self.width}x{self.height};K={k};R={','.join(f'{v:.4f}' for v in self.R)}"

    @classmethod
    def parse(cls, text: str) -> "RoiSignature":
        pad, max_side, size, k, r = text.split(";")
        width, height = size.split("x")
        side: str = max_side.removeprefix("max_side=")
        return cls.of(
            float(pad.removeprefix("pad=")), None if side == "None" else int(side), int(width), int(height), floats(k.removeprefix("K=")), floats(r.removeprefix("R="))
        )


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PriorLayerProperties:
    """The ``moge2_prior`` layer's property: model, source, window, sizes, the reader that made the clip and the camera grid."""

    model: str
    predictor: str
    resolution_level: int
    fov_source: str
    source: str
    frame_start: int
    frame_end: int
    num_frames: int
    max_side: int
    """0 = the stored size."""
    cameras: tuple[str, ...]
    image_sizes: tuple[tuple[int, int], ...]
    loader: str
    roi: str
    """``none`` or ``people pad=<pad> max_side=<cap>`` (for people; the grid itself is ``roi_cameras``)."""
    roi_cameras: dict[str, RoiSignature | None]
    """Per camera: None for a real camera, else its ROI grid."""

    def to_properties(self) -> dict[str, PropertyValue]:
        return {
            "model": self.model,
            "predictor": self.predictor,
            "resolution_level": self.resolution_level,
            "fov_source": self.fov_source,
            "source": self.source,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "num_frames": self.num_frames,
            "max_side": self.max_side,
            "cameras": ",".join(self.cameras),
            "image_sizes": ",".join(f"{w}x{h}" for w, h in self.image_sizes),
            "loader": self.loader,
            "roi": self.roi,
            "roi_cameras": encode_roi_cameras(self.roi_cameras),
        }


def encode_roi_cameras(cameras: Mapping[str, RoiSignature | None]) -> str:
    """``name@<signature>|...`` (``none`` for a real camera)."""
    return "|".join(f"{name}@{'none' if sig is None else sig.encode()}" for name, sig in cameras.items())


def parse_roi_cameras(text: str | None) -> dict[str, RoiSignature | None]:
    """The ``roi_cameras`` cell of a prior layer (None or empty: a layer written before ROI cameras, every camera real)."""
    out: dict[str, RoiSignature | None] = {}
    for item in text.split("|") if text else ():
        name, sig = item.split("@", 1)
        out[name] = None if sig == "none" else RoiSignature.parse(sig)
    return out


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MatteLayerProperties:
    """The ``people_matte`` layer's property."""

    model: str
    frame_start: int
    num_frames: int
    source_catalog: str
    source_dataset: str
    source_segment: str
    crop: str
    """How the person box was made (``smplx box pad ...``, ``projected joints ...``)."""
    time_start_s: float | None = None
    """SelfCap segments: the clip's start on ``video_time``."""

    def to_properties(self) -> dict[str, PropertyValue]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class InitCloudProperties:
    """The ``init_cloud`` property beside a logged init cloud (``init_cloud.log_init_cloud``)."""

    entity: str
    frame_start: int
    frame_end: int
    keyframe_step: int
    fps: float
    source_frame_start: int
    dataset: str
    segment_id: str
    method: str
    num_rows: int

    def to_properties(self) -> dict[str, PropertyValue]:
        return dataclasses.asdict(self)

    @classmethod
    def from_cells(cls, cells: Cells) -> "InitCloudProperties":
        values: dict[str, object] = {k: one(cells, k) for k in cells}
        if isinstance(values.get("fps"), int):
            values["fps"] = float(values["fps"])  # type: ignore[arg-type]
        return from_dict(cls, values)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class InitRomaProperties:
    """The ``init_roma`` layer's property (``open4dgs-init-roma``)."""

    holdout: tuple[str, ...]
    init_cameras: tuple[str, ...]
    num_holdout: int
    holdout_seed: int
    max_side: int
    masks: str
    camera_neighbors: int
    wide_baseline_per_camera: int
    roma_v2_setting: str
    memfof_max_side: int
    subkeyframe_min_speed: float
    """-1 = off."""
    extra_args: str
    init_seconds: float

    def to_properties(self) -> dict[str, PropertyValue]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}
