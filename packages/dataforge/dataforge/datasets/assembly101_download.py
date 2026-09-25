"""Driver-only transport: range-read selected zip members into local staging."""

import shutil
from pathlib import Path
from typing import BinaryIO, cast
from zipfile import ZipFile

from huggingface_hub import HfFileSystem

from dataforge.datasets.assembly101_source import NAS_ROOT, POSE_MEMBERS

MIRROR_ZIP: str = "datasets/pablovela5620/assembly101-720p@001839131530cee9b2deb9ca66c025998d10cba4/AssemblyPoses.zip"


def fetch_pose_members(destination: Path, sequences: tuple[str, ...], members: tuple[str, ...] = POSE_MEMBERS) -> None:
    """Fetch selected members only; existing files with the exact zip size are skipped.

    This opt-in network function is never called by download(), conversion or tests.
    The driver owns its staging destination and subsequent raw-tree placement.
    """
    if destination.resolve().is_relative_to(NAS_ROOT):
        raise ValueError("Assembly101 extraction must use local staging, never the NAS")
    if not sequences or any("/" in name or name in (".", "..") for name in (*sequences, *members)):
        raise ValueError("explicit plain sequence and member names are required")
    filesystem: HfFileSystem = HfFileSystem()
    with filesystem.open(MIRROR_ZIP, "rb", block_size=16 * 1024 * 1024) as remote, ZipFile(cast(BinaryIO, remote)) as archive:
        for sequence in sequences:
            for member in members:
                name: str = f"assembly101_camera_and_hand_poses/{member}/{sequence}.json"
                info = archive.getinfo(name)
                target: Path = destination / name
                if target.is_file() and target.stat().st_size == info.file_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary: Path = target.with_suffix(".json.part")
                try:
                    with archive.open(info) as source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                    if temporary.stat().st_size != info.file_size:
                        raise ValueError(f"incomplete member: {name}")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
