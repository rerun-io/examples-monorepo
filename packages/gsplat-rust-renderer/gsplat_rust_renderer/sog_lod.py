"""Streamed SOG reader: `lod-meta.json` (PlayCanvas / splat-transform LOD writer, version 1) and its SOG files.

Layout as splat-transform v3.3 writes it: `lod-meta.json` at the root, one SOG directory per file
(`<level>_<chunk>/meta.json` + WebP planes), an optional `environment` SOG (drawn at every level).
The tree's leaves hold one contiguous run per level: `lods[level] = {file, offset, count}` (rows of that file).
Third-party schema: unknown fields are ignored; only what the viewer needs is typed here.
"""

from dataclasses import dataclass
from pathlib import Path

from serde import SerdeError, serde
from serde.core import coerce
from serde.json import from_json

from gsplat_rust_renderer.sog_decode import SogFile, read_sog

LOD_META: str = "lod-meta.json"


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class BoundMeta:
    min: list[float]
    max: list[float]


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class RunMeta:
    file: int
    offset: int
    count: int


@dataclass(frozen=True, slots=True)
class NodeMeta:
    """A tree node: inner nodes have `children`, leaves have `lods` (level as a decimal string -> run)."""

    bound: BoundMeta
    children: list["NodeMeta"] | None = None
    lods: dict[str, RunMeta] | None = None


serde(NodeMeta, type_check=coerce)


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class LodMeta:
    """lod-meta.json version 1 (third-party schema: unknown fields are ignored)."""

    version: int
    filenames: list[str]
    tree: NodeMeta
    lodLevels: int | None = None
    counts: list[int] | None = None
    environment: str | None = None

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"lod-meta version {self.version} is not supported (1 only)")


@dataclass(frozen=True, slots=True)
class LodRun:
    """Rows [offset, offset + count) of file `file` hold one leaf at one level."""

    level: int
    """0 = finest."""
    file: int
    """Index into `SogLodSet.files` (the lod-meta `filenames` order)."""
    offset: int
    count: int


@dataclass(frozen=True, slots=True)
class LodLeaf:
    bound_min: tuple[float, float, float]
    """Axis-aligned box of the leaf in the file frame."""
    bound_max: tuple[float, float, float]
    runs: list[LodRun]
    """One run per level, finest first."""


@dataclass(frozen=True, slots=True)
class SogLodSet:
    """A Streamed SOG as stored: lod-meta.json text and every SOG file (levels first, environment last)."""

    lod_meta_json: str
    files: list[str]
    """Relative meta.json paths: `filenames`, then `environment` when present."""
    sogs: list[SogFile]
    """Same order as `files`."""
    file_levels: list[int | None]
    """Level of each file's rows; None for the environment."""
    levels: int
    counts: list[int]
    """Rows per level (lod-meta `counts`; empty when absent). Reader facts for checks and tests; the writer does
    not use them."""
    leaves: list[LodLeaf]
    """The tree's leaves as parsed for the run checks of `read_sog_lod`; the writer does not use them."""

    @property
    def file_counts(self) -> list[int]:
        return [sog.meta.count for sog in self.sogs]


def _leaves(node: NodeMeta, out: list[LodLeaf]) -> None:
    if node.lods is not None:
        runs: list[LodRun] = [
            LodRun(int(level), run.file, run.offset, run.count) for level, run in sorted(node.lods.items(), key=lambda kv: int(kv[0]))
        ]
        low: list[float] = node.bound.min
        high: list[float] = node.bound.max
        out.append(LodLeaf((low[0], low[1], low[2]), (high[0], high[1], high[2]), runs))
    for child in node.children or []:
        _leaves(child, out)


def read_sog_lod(path: Path) -> SogLodSet:
    """A Streamed SOG from its `lod-meta.json` or the directory holding it; raises ValueError naming the file."""
    meta_path: Path = path / LOD_META if path.is_dir() else path
    text: str = meta_path.read_text()
    try:
        meta: LodMeta = from_json(LodMeta, text)
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{meta_path}: not a Streamed SOG lod-meta.json version 1 ({error})") from error
    files: list[str] = list(meta.filenames) + ([meta.environment] if meta.environment is not None else [])
    sogs: list[SogFile] = [read_sog(meta_path.parent / f) for f in files]
    leaves: list[LodLeaf] = []
    _leaves(meta.tree, leaves)
    file_levels: list[int | None] = [None] * len(files)
    for leaf in leaves:
        for run in leaf.runs:
            if not 0 <= run.file < len(meta.filenames):
                raise ValueError(f"{meta_path}: a level {run.level} run names file {run.file}, outside the filenames")
            if run.offset + run.count > sogs[run.file].meta.count:
                raise ValueError(
                    f"{meta_path}: a level {run.level} run [{run.offset}, {run.offset + run.count}) reaches past the "
                    f"{sogs[run.file].meta.count} splats of {files[run.file]}"
                )
            if file_levels[run.file] not in (None, run.level):
                raise ValueError(f"{meta_path}: {files[run.file]} holds rows of levels {file_levels[run.file]} and {run.level}")
            file_levels[run.file] = run.level
    levels: int = meta.lodLevels if meta.lodLevels is not None else 1 + max((r.level for leaf in leaves for r in leaf.runs), default=0)
    counts: list[int] = list(meta.counts) if meta.counts is not None else []
    return SogLodSet(text, files, sogs, file_levels, levels, counts, leaves)
