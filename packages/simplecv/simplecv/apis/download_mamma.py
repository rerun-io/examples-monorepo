"""Download one MAMMA sequence per subset from the MPI download gateway.

Register at https://mamma.is.tue.mpg.de/register.php first, then export the
credentials (the gateway rejects anonymous requests with HTTP 401)::

    export MAMMA_USERNAME='your_email'
    export MAMMA_PASSWORD='your_password'
    pixi run -e simplecv simplecv-download-mamma

By default this fetches ONE sequence from each subset (dance, multi-people,
iphone, eval, syn) with the smallest video variant (``videos_crf24`` for IOI
rigs, ``videos_light`` for iPhones — both H.265 yuv444, so run
``simplecv-preprocess-mamma`` afterwards to build the AV1 yuv420 mirror).
Existing valid files are skipped, so reruns only fetch what is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests
from tqdm import tqdm

BASE_URL: str = "https://download.is.tue.mpg.de/download.php?domain=mamma&resume=1"
REGISTER_URL: str = "https://mamma.is.tue.mpg.de/register.php"

IOI_CAMERAS_32: tuple[str, ...] = tuple(f"IOI_{i:02d}" for i in range(1, 33))
IOI_CAMERAS_16: tuple[str, ...] = IOI_CAMERAS_32[:16]
IPHONE_CAMERAS: tuple[str, ...] = ("A001", "B001", "C001", "D001")

MammaSubset = Literal["dance", "multi-people", "iphone", "eval", "syn"]


@dataclass
class SubsetSpec:
    """Remote layout of one MAMMA subset's default sequence."""

    sequence: str
    """Sequence path relative to the remote ``datasets/`` root (and the local output dir)."""
    cameras: tuple[str, ...]
    """Camera names (`meta|gt/<cam>.npz` + `<video_dir>/<cam>.mp4`)."""
    video_dir: str
    """Remote video variant to fetch (`videos_crf24` for IOI, `videos_light` for iPhone)."""
    calib_dir: Literal["meta", "gt"]
    """Calibration dir: `meta` (markerless, with `pred/`) or `gt` (eval)."""
    num_people: int
    """Number of `pred/params_XX.npz` files (0 for eval subsets, which ship GT in `gt/global.npz`)."""


DEFAULT_SUBSET_SPECS: dict[str, SubsetSpec] = {
    "dance": SubsetSpec(
        sequence="mamma_markerless_dance/050825_WestCoastSwing_CutOff_03688_03689_1",
        cameras=IOI_CAMERAS_32,
        video_dir="videos_crf24",
        calib_dir="meta",
        num_people=2,
    ),
    "multi-people": SubsetSpec(
        sequence="mamma_markerless_multiple_people/260216_MultiMama_3_accidental_bump_000111_1",
        cameras=IOI_CAMERAS_32,
        video_dir="videos_crf24",
        calib_dir="meta",
        num_people=3,
    ),
    "iphone": SubsetSpec(
        sequence="mamma_markerless_iphones/indoors/crossing_arms",
        cameras=IPHONE_CAMERAS,
        video_dir="videos_light",
        calib_dir="meta",
        num_people=1,
    ),
    "eval": SubsetSpec(
        sequence="mamma_eval_singles/230929_WhiteRabbit_CatchBall_50048_1",
        cameras=IOI_CAMERAS_16,
        video_dir="videos_crf24",
        calib_dir="gt",
        num_people=0,
    ),
}


@dataclass
class DownloadConfig:
    """Configuration for the single-sequence-per-subset MAMMA download."""

    output_dir: Path = Path("data/mamma")
    """Local root; files land at ``<output_dir>/<subset>/<sequence>/...``."""
    subsets: tuple[MammaSubset, ...] = ("dance", "multi-people", "iphone", "eval", "syn")
    """Subsets to fetch (one default sequence each)."""
    username: str | None = None
    """MAMMA account username; falls back to the MAMMA_USERNAME env var."""
    password: str | None = None
    """MAMMA account password; falls back to the MAMMA_PASSWORD env var."""
    max_cameras: int | None = None
    """Optional cap on cameras per sequence for quick tests (None = all)."""
    syn_dataset: str = "moyo_4-6_C_200_00"
    """MammaSyn WebDataset to sample from (synthetic training data)."""
    max_syn_shards: int = 1
    """Number of WebDataset shards to fetch from the syn manifest."""


def _looks_like_error_payload(head: bytes) -> bool:
    """Detect the gateway's HTML/error bodies served instead of a real file."""
    head_stripped: bytes = head.lstrip()[:64].lower()
    return head_stripped.startswith((b"error:", b"<!doctype html", b"<html"))


def is_valid_download(path: Path) -> bool:
    """True when ``path`` exists, is non-empty, and is not a gateway error page."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        return not _looks_like_error_payload(f.read(256))


def download_mamma_file(sfile: str, dest_path: Path, *, username: str, password: str) -> bool:
    """Fetch one remote file via credentialed POST; returns False when missing remotely.

    Args:
        sfile: Remote path passed as the gateway's ``sfile=`` query value
            (e.g. ``datasets/<sequence>/meta/global.npz``).
        dest_path: Local destination; skipped when already valid.
        username: MAMMA account username.
        password: MAMMA account password.

    Returns:
        True when the file exists locally and is valid after the call.

    Raises:
        RuntimeError: On authentication failure (HTTP 401).
    """
    if is_valid_download(dest_path):
        print(f"  [skip] {sfile}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path = dest_path.with_name(f"{dest_path.name}.part")
    url: str = f"{BASE_URL}&sfile={sfile}"
    try:
        response: requests.Response = requests.post(
            url,
            data={"username": username, "password": password},
            stream=True,
            timeout=(30, 300),
        )
        if response.status_code == 401:
            raise RuntimeError(f"MAMMA credentials rejected (HTTP 401). Register at {REGISTER_URL} and set MAMMA_USERNAME/MAMMA_PASSWORD.")
        if not response.ok:
            print(f"  [FAIL] {sfile} (HTTP {response.status_code})")
            return False

        total_size: int = int(response.headers.get("content-length", 0))
        first_chunk: bool = True
        with open(tmp_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name, leave=False) as pbar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if first_chunk and _looks_like_error_payload(chunk):
                    tmp_path.unlink(missing_ok=True)
                    print(f"  [missing] {sfile}")
                    return False
                first_chunk = False
                f.write(chunk)
                pbar.update(len(chunk))
    except requests.RequestException as exc:
        # Isolate transient network failures per file so one drop mid-run does
        # not abort the remaining files/subsets (rerun resumes the [FAIL]s).
        tmp_path.unlink(missing_ok=True)
        print(f"  [FAIL] {sfile} ({type(exc).__name__}: {exc})")
        return False

    if not is_valid_download(tmp_path):
        tmp_path.unlink(missing_ok=True)
        print(f"  [FAIL] {sfile} (empty or error payload)")
        return False
    tmp_path.replace(dest_path)
    print(f"  [ok] {sfile}")
    return True


def sequence_files_for_spec(spec: SubsetSpec, max_cameras: int | None) -> list[str]:
    """Per-sequence remote file list (relative to the sequence dir)."""
    cameras: tuple[str, ...] = spec.cameras[:max_cameras] if max_cameras is not None else spec.cameras
    files: list[str] = [f"{spec.calib_dir}/global.npz"]
    files += [f"{spec.calib_dir}/{cam}.npz" for cam in cameras]
    files += [f"pred/params_{person_idx:02d}.npz" for person_idx in range(spec.num_people)]
    files += [f"{spec.video_dir}/{cam}.mp4" for cam in cameras]
    return files


def download_subset(spec: SubsetSpec, config: DownloadConfig, *, username: str, password: str) -> tuple[int, int]:
    """Download one subset's default sequence; returns (num_ok, num_total)."""
    files: list[str] = sequence_files_for_spec(spec, config.max_cameras)
    num_ok: int = 0
    for rel_file in files:
        ok: bool = download_mamma_file(
            f"datasets/{spec.sequence}/{rel_file}",
            config.output_dir / spec.sequence / rel_file,
            username=username,
            password=password,
        )
        num_ok += int(ok)
    return num_ok, len(files)


def download_syn_shards(config: DownloadConfig, *, username: str, password: str) -> tuple[int, int]:
    """Download the syn manifest plus the first ``max_syn_shards`` WebDataset shards.

    MammaSyn is served under ``datasets/training_webdataset/`` and is flagged
    "coming soon" upstream, so missing files are tolerated (not an error).
    """
    local_root: Path = config.output_dir / "mamma_syn_wd" / config.syn_dataset
    remote_root: str = f"datasets/training_webdataset/{config.syn_dataset}"
    manifest_path: Path = local_root / "tar_train_list.txt"
    if not download_mamma_file(f"{remote_root}/tar_train_list.txt", manifest_path, username=username, password=password):
        print(f"  [warn] MammaSyn manifest unavailable for {config.syn_dataset} (upstream marks syn data 'coming soon')")
        return 0, 1

    shard_names: list[str] = [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]
    shard_names = shard_names[: config.max_syn_shards]
    num_ok: int = 1
    for shard in shard_names:
        ok: bool = download_mamma_file(f"{remote_root}/{shard}", local_root / shard, username=username, password=password)
        num_ok += int(ok)
    return num_ok, 1 + len(shard_names)


def main(config: DownloadConfig) -> None:
    """Download one MAMMA sequence per requested subset."""
    username: str | None = config.username or os.environ.get("MAMMA_USERNAME")
    password: str | None = config.password or os.environ.get("MAMMA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            f"MAMMA credentials required: register at {REGISTER_URL}, then set MAMMA_USERNAME and MAMMA_PASSWORD "
            "(or pass --username/--password)."
        )

    totals: dict[str, tuple[int, int]] = {}
    for subset in config.subsets:
        print(f"\n=== {subset} ===")
        if subset == "syn":
            totals[subset] = download_syn_shards(config, username=username, password=password)
        else:
            totals[subset] = download_subset(DEFAULT_SUBSET_SPECS[subset], config, username=username, password=password)

    print(f"\nDone. Files under {config.output_dir}:")
    for subset, (num_ok, num_total) in totals.items():
        print(f"  {subset}: {num_ok}/{num_total} files present")
    if any(subset != "syn" for subset in config.subsets):
        print("Next: pixi run -e simplecv simplecv-preprocess-mamma  (AV1 yuv420 re-encode for the fast decode path)")
