"""EPFL raw tree from its pinned HuggingFace mirror: listing, per-session files, verified resumable fetch."""

import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from dataforge.datasets.epfl_source import CAMERA_NAMES
from dataforge.transports import HfFileInfo, IntegrityError, publish_verified

SOURCE_REPO: str = "pablovela5620/epfl-smart-kitchen-av1"
"""AV1 mirror of the public release: videos, meta data, poses and annotations (pinned at SOURCE_REVISION)."""
SMPL_REPO: str = "pablovela5620/mamma-streaming-data"
"""Private dataset holding the official neutral SMPL model; contacted only when the model is not already in place."""
SMPL_REVISION: str = "e05c6b723e24e2c0fc79201e06e8c8fbaa6cedfd"
SMPL_FILE: str = "body_models/smpl/SMPL_NEUTRAL.pkl"
POSE_DIR: str = "Public_release_pose"
VIDEO_DIR: str = "Public_release_videos"


def session_files(key: str) -> tuple[str, ...]:
    """Repo-relative files one session's conversion reads (depth videos, IMUs and depth timestamps are not ingested)."""
    pose = f"{POSE_DIR}/{key}"
    video = f"{VIDEO_DIR}/{key}"
    return (
        f"{pose}/pose_3d/pose3d_mano.csv",
        f"{pose}/pose_3d/pose3d_smpl.csv",
        f"{pose}/annotations/actions_annotations.xlsx",
        f"{pose}/annotations/activity_annotations.json",
        *(f"{video}/videos/{name}.mp4" for name in CAMERA_NAMES),
        *(f"{video}/meta_data/{name}" for name in ("camera_matrix.json", "timestamps.txt", "holo_data_wpose.csv")),
    )


def complete_sessions(listing: list[HfFileInfo]) -> dict[str, list[HfFileInfo]]:
    """Group the listing into split/subject/session keys that ship every file conversion needs."""
    by_path = {file.path: file for file in listing}
    keys = sorted({"/".join(file.path.split("/")[1:4]) for file in listing if file.path.startswith(f"{POSE_DIR}/") and file.path.count("/") >= 4})
    sessions: dict[str, list[HfFileInfo]] = {}
    for key in keys:
        required = session_files(key)
        absent = [path for path in required if path not in by_path]
        if absent:
            print(f"epfl: {key} is incomplete on the Hub ({len(absent)} files absent, first {absent[0]}); not listed", file=sys.stderr)
            continue
        sessions[key] = [by_path[path] for path in required]
    return sessions


class FetchError(ValueError):
    """One file could not be fetched or verified; download() collects these and raises once at the end."""


def fetch(repo_id: str, revision: str, file: HfFileInfo, dest: Path, staging: Path) -> bool:
    """Fetch one file to dest unless it already has the listed size; return whether bytes moved.

    The Hub client writes into staging, never onto dest. It keeps no resumable partial: an interrupted file is
    discarded and fetched again whole. Only a copy whose size and hash match the listing is renamed onto dest
    (``publish_verified``: sha256 for LFS files, the git blob sha1 otherwise), so a failed fetch leaves any old
    dest intact and a dest of the listed size was verified when it landed. A mismatching copy is set aside
    under a unique ``.mismatch`` name in staging and raises FetchError.
    """
    if dest.is_file() and dest.stat().st_size == file.size_bytes:
        return False
    # Temp files of killed runs have process-unique names and are never reused.
    for stale in (staging / ".cache/huggingface/download" / file.path).parent.glob("*.incomplete"):
        stale.unlink()
    try:
        landed = Path(hf_hub_download(repo_id, file.path, repo_type="dataset", revision=revision, local_dir=str(staging), force_download=True))
    except HfHubHTTPError as error:
        # Status only, no chained traceback: the error text can carry a signed redirect URL.
        raise FetchError(f"{repo_id}@{revision[:8]}:{file.path}: HTTP {error.response.status_code}") from None
    try:
        publish_verified(landed, dest, file.integrity)
    except IntegrityError:
        quarantine = landed.with_name(f"{landed.name}.{time.time_ns()}.mismatch")
        landed.rename(quarantine)
        raise FetchError(f"{quarantine}: size or hash differs from {repo_id}@{revision[:8]}:{file.path} ({file.size_bytes} bytes); rerun") from None
    return True
