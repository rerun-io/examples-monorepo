"""HOT3D name matching and asset preparation.

Rerun 0.38.1 rejects glTF files requiring KHR_texture_transform
(checked with native Viewer pixels on 2026-09-25).
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import NamedTuple

from serde import serde

from dataforge import transports
from dataforge.datasets.show3d_source import OBJECTS, read_json
from dataforge.objects import strip_texture_transform

MESH_REPO: str = "bop-benchmark/hot3d"


@serde
@dataclass(frozen=True, slots=True)
class MeshInfo:
    """Only names are needed from the third-party BOP census."""

    name: str
    """HOT3D object name, independent of BOP numbering."""


@cache
def mesh_ids(asset_root: Path) -> dict[str, int]:
    """Read the BOP census once and resolve all mapped aliases by name."""
    path: Path = asset_root / "object_models/models_info.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run dataforge-download show3d first")
    records: dict[str, MeshInfo] = read_json(path, dict[str, MeshInfo])
    try:
        by_name: dict[str, int] = {record.name: int(key) for key, record in records.items()}
        return {alias: by_name[name] for alias, name in OBJECTS.items() if name is not None}
    except (ValueError, KeyError) as error:
        raise ValueError(f"{path}: missing model name or non-numeric model ID: {error}") from error


class MeshAsset(NamedTuple):
    """Resolved BOP identity and ready-to-log GLB."""

    mesh_id: int
    path: Path


def stripped_mesh(raw_root: Path, alias: str) -> MeshAsset:
    """Resolve one ready-to-log asset, naming the download command if absent."""
    root: Path = raw_root / "assets/hot3d_bop"
    mesh_id: int = mesh_ids(root)[alias]
    path: Path = root / f"stripped/obj_{mesh_id:06d}.glb"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run dataforge-download show3d first")
    return MeshAsset(mesh_id, path)


def download_meshes(raw_root: Path) -> None:
    """Fetch the BOP census and all mapped assets; strip each mesh once."""
    root: Path = raw_root / "assets/hot3d_bop"
    revision: str | None = transports.repo_revision(MESH_REPO, None)
    if revision is None:
        raise RuntimeError(f"{MESH_REPO}: no commit SHA")
    info: str = "object_models/models_info.json"
    if transports.local_verify(root, required=[info]):
        transports.hf_fetch_files(MESH_REPO, [info], local_dir=root, revision=revision)
    ids: dict[str, int] = mesh_ids(root)
    (root / "stripped").mkdir(parents=True, exist_ok=True)
    for mesh_id in sorted(set(ids.values())):
        name: str = f"object_models/obj_{mesh_id:06d}.glb"
        raw: Path = root / name
        target: Path = root / f"stripped/obj_{mesh_id:06d}.glb"
        if not target.is_file():
            if not raw.is_file():
                transports.hf_fetch_files(MESH_REPO, [name], local_dir=root, revision=revision)
            temporary: Path = target.with_suffix(".glb.tmp")
            temporary.write_bytes(strip_texture_transform(raw.read_bytes()))
            temporary.replace(target)
        raw.unlink(missing_ok=True)
