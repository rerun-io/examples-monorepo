"""MAMMA download: verify the release in place, and prove the MPI fetch on chosen sequences.

The release (~221 GB of MPI captures) is already on the NAS, verified complete on 2026-07-29 by ``mamma-download-dataset``,
so dataforge never re-downloads it. ``verify_release`` checks what the converter reads, without network. ``fetch_sequence``
downloads one sequence's files from the MPI server into a scratch directory and compares each with the release copy: the
proof that the download path still works. The MPI server takes the account (``MAMMA_USERNAME``, ``MAMMA_PASSWORD``) as a POST
body; it served Python clients HTML in July 2026 and serves them the files in October 2026.
"""

import hashlib
import os
from pathlib import Path
from urllib.parse import quote

import requests

from dataforge.datasets.mamma_source import FAMILY_COLLECTIONS, Family, MammaSequence, discover_sequences
from dataforge.transports import CHUNK_BYTES

MPI_URL: str = "https://download.is.tue.mpg.de/download.php?domain=mamma&resume=1"
EXPECTED_SEQUENCES: dict[str, int] = {
    "mamma_eval_singles": 22,
    "mamma_eval_extra": 12,
    "mamma_eval_dance": 18,
    "mamma_markerless_dance": 123,
    "mamma_markerless_multiple_people": 34,
    "mamma_markerless_iphones": 42,
}
"""Sequences of each collection in the release manifest (``packages/mamma``'s vendored copy of the MPI download scripts)."""
ERROR_MARKERS: tuple[bytes, ...] = (b"error: file not found.", b"<!doctype html", b"<html")
"""What the MPI server sends instead of a file it refuses: its error text or its login page."""
TIMEOUT_S: float = 120.0


def sequence_files(sequence: MammaSequence) -> list[Path]:
    """Every release file the converter reads for ``sequence``."""
    files: list[Path] = [sequence.labels_dir / "global.npz", *sequence.predictions]
    for camera in sequence.cameras:
        files += [sequence.calibration(camera), sequence.video(camera)]
        if sequence.collection.startswith("mamma_eval_"):
            files.append(sequence.mask_tar(camera))
    if sequence.markers_dir.is_dir():
        files += [sequence.markers_dir / name for name in ("labels_m37.npy", "vicon_m37.npy", "baseline_m37.npy")]
    return files


def file_problem(path: Path) -> str | None:
    """Why ``path`` cannot be used: missing or unreadable, empty, or an MPI error page saved as the file; None when fine."""
    try:
        with path.open("rb") as stream:
            prefix: bytes = stream.read(256).lower()
    except OSError as error:
        return f"{path}: {type(error).__name__}"
    if not prefix:
        return f"{path}: empty"
    if any(marker in prefix for marker in ERROR_MARKERS):
        return f"{path}: an HTML page or an MPI error, not the file"
    return None


def verify_release(root: Path, family: Family, *, expected: dict[str, int] = EXPECTED_SEQUENCES) -> tuple[list[MammaSequence], list[str]]:
    """The family's convertible sequences under ``root``, and every problem found: skipped directories, partial rigs, bad files,
    missing sequences. Opens every file the converter reads and reads 256 bytes of it; nothing is written."""
    sequences, skipped = discover_sequences(root, family)
    problems: list[str] = [reason for reason in skipped if "bachata_improv_2p_32cam" not in reason]  # a CRF 24 variant tree without labels
    for sequence in sequences:
        problems += [problem for path in sequence_files(sequence) if (problem := file_problem(path)) is not None]
    for collection in FAMILY_COLLECTIONS[family]:
        found: int = sum(sequence.collection == collection for sequence in sequences)
        if found != expected.get(collection, found):
            problems.append(f"{collection}: {found} sequences, the release has {expected[collection]}")
    return sequences, problems


def post_to_file(url: str, data: dict[str, str], target: Path) -> None:
    """POST ``data`` to ``url`` and stream the response into ``target``."""
    with requests.post(url, data=data, stream=True, timeout=TIMEOUT_S) as response:
        response.raise_for_status()
        with target.open("wb") as stream:
            for block in response.iter_content(CHUNK_BYTES):
                stream.write(block)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def fetch_sequence(root: Path, sequence: MammaSequence, into: Path) -> list[str]:
    """Download ``sequence``'s files from the MPI server under ``into`` (release-relative paths) and compare each with the release
    copy under ``root``; return the files that differ. Each lands as ``.part`` and is renamed once complete."""
    username: str | None = os.environ.get("MAMMA_USERNAME")
    password: str | None = os.environ.get("MAMMA_PASSWORD")
    if not username or not password:
        raise RuntimeError("the MPI fetch needs MAMMA_USERNAME and MAMMA_PASSWORD (a https://mamma.is.tue.mpg.de account) in the environment")
    mismatches: list[str] = []
    for path in sequence_files(sequence):
        relative: str = path.relative_to(root).as_posix()
        target: Path = into / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        part: Path = target.with_name(f"{target.name}.part")
        post_to_file(f"{MPI_URL}&sfile={quote(f'datasets/{relative}', safe='/')}", {"username": username, "password": password}, part)
        problem: str | None = file_problem(part)
        if problem is not None:
            part.unlink()
            raise RuntimeError(f"MPI fetch of {relative}: {problem.removeprefix(f'{part}: ')}")
        part.rename(target)
        if target.stat().st_size != path.stat().st_size or sha256(target) != sha256(path):
            mismatches.append(f"{relative}: {target.stat().st_size} bytes fetched differ from the release copy")
    return mismatches
