"""Shared download utilities for CDN-hosted datasets."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, dest_path: Path, expected_size: int | None = None) -> None:
    """Download a file with progress bar, skipping if already complete."""
    if dest_path.exists() and expected_size is not None and dest_path.stat().st_size == expected_size:
        return  # Already downloaded
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response: requests.Response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total_size: int = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=dest_path.name,
        leave=False,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def verify_sha1(file_path: Path, expected_sha1: str) -> bool:
    """Verify SHA1 checksum of a downloaded file."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha1.update(chunk)
    return sha1.hexdigest() == expected_sha1


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP file and remove the archive.

    Validates member paths to prevent Zip Slip (path traversal).
    """
    extract_root: Path = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path: Path = (extract_root / member.filename).resolve()
            if extract_root not in member_path.parents and member_path != extract_root:
                raise ValueError(f"Unsafe ZIP member path {member.filename!r} in {zip_path}")
        zf.extractall(extract_dir)
    zip_path.unlink()
