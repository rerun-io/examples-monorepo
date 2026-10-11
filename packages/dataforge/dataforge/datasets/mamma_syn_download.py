"""MammaSyn download: verify the NAS copy against the pinned Hub listing, and prove a fetch on chosen sequences.

The release is the gated Hugging Face dataset ``Intelligent-Systems/MammaSyn`` (6.63 TB). It is never downloaded again:
``download`` lists the pinned revision and checks that every shard and ``metadata.json`` of the family is on disk at its
listed size. With ``--fetch-into``, it fetches the shards holding the chosen sequences and compares them byte for byte.
"""

import filecmp
from pathlib import Path

from dataforge import transports
from dataforge.datasets.mamma_syn_source import BEDLAM_DATASET, SYN_DIR, SynFamily
from dataforge.transports import HfFileInfo

MAMMASYN_REPO: str = "Intelligent-Systems/MammaSyn"
MAMMASYN_REVISION: str = "01ecdc887819a91e992b58fdb2c1684335543558"
"""The commit the NAS copy was downloaded at (its Hub cache tree names it; ``main`` on 2026-10-11)."""


def family_files(listing: list[HfFileInfo], family: SynFamily) -> list[HfFileInfo]:
    """The shards and ``metadata.json`` files of ``family`` in a listing (the training lists are not read)."""
    return [
        info
        for info in listing
        if (info.path.endswith(".tar") or info.path.endswith("/metadata.json")) and (family == "syn-bedlam") == info.path.startswith(f"{BEDLAM_DATASET}/")
    ]


def verify_syn(root: Path, family: SynFamily, listing: list[HfFileInfo]) -> tuple[int, list[str]]:
    """How many of the family's listed files there are, and every one missing under ``root/MammaSyn`` or of another size."""
    files: list[HfFileInfo] = family_files(listing, family)
    problems: list[str] = []
    for info in files:
        local: Path = root / SYN_DIR / info.path
        if not local.is_file():
            problems.append(f"{local}: missing")
        elif (size := local.stat().st_size) != info.size_bytes:
            problems.append(f"{local}: {size} bytes, the Hub lists {info.size_bytes}")
    return len(files), problems


def fetch_proof(root: Path, directory: Path, shards: list[str], into: Path) -> list[str]:
    """Fetch ``shards`` of a sequence or batch ``directory`` and its ``metadata.json`` from the Hub into ``into``; every file that
    differs from the release copy under ``root``."""
    relative: Path = directory.relative_to(root / SYN_DIR)
    paths: list[str] = [f"{relative}/{name}" for name in [*shards, "metadata.json"]]
    transports.hf_fetch_files(MAMMASYN_REPO, paths, local_dir=into, revision=MAMMASYN_REVISION)
    return [f"{path}: differs from the release copy" for path in paths if not filecmp.cmp(into / path, root / SYN_DIR / path, shallow=False)]
