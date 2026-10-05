"""Ego-Exo4D raw tree: HM fits and body models from pinned Hugging Face commits, take files from the Ego-Exo4D S3 release.

The release needs the Ego-Exo4D licence keys (https://ego4ddataset.com), read as an AWS profile. Its manifest
(``<release>/<part>/manifest.json``, the file the official ``egoexo`` CLI reads) lists every file of a part per
take or capture uid with its size, so dataforge fetches exactly the files a take's conversion reads and nothing
else, verified by size, into the layout the CLI writes (``<root>/<relative_path>``).
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
from serde import serde
from serde.json import from_json

from dataforge import transports
from dataforge.datasets.egoexo4d_body import HM_FILE, HM_REPO, HM_REVISION, MODEL_REPO, MODEL_REVISION, SMPLH_FILE, SMPLX_FILE
from dataforge.transports import FileIntegrity, HfFileInfo

RELEASE: str = "s3://ego4d-consortium-sharing/egoexo-public/v2"
"""The public Ego-Exo4D release the official ``egoexo`` CLI downloads (``--release v2``)."""
HM_DIR: str = "hm"
"""Raw-root subdirectory holding ``<take_name>/<HM_FILE>``."""
Part: TypeAlias = Literal["metadata", "takes", "take_trajectory", "captures", "take_vrs_noimagestream"]
"""Ego-Exo4D release parts this dataset reads."""


def hm_fits() -> dict[str, HfFileInfo]:
    """Every take the pinned HM release ships a fit for, keyed by take name; one recursive listing call."""
    entries = HfApi().list_repo_tree(HM_REPO, repo_type="dataset", revision=HM_REVISION, recursive=True)
    return {
        entry.path.split("/")[0]: transports.hf_file_info(entry)
        for entry in entries
        if isinstance(entry, RepoFile) and entry.path.endswith(f"/{HM_FILE}") and entry.path.count("/") == 1
    }


def fit_path(root: Path, take: str) -> Path:
    """Where download() puts one take's fit."""
    return root / HM_DIR / take / HM_FILE


def fetch_fit(root: Path, file: HfFileInfo) -> bool:
    """Fetch one take's fit to ``root/hm`` unless it is already there, verified by its LFS sha256."""
    return transports.hf_fetch_verified(HM_REPO, file, local_dir=root / HM_DIR, revision=HM_REVISION)


def fetch_models(model_root: Path) -> int:
    """Fetch the SMPL-H male and SMPL-X neutral models unless already in place; return how many were fetched.

    The model repo is private: without access, place the two files at ``model_root/<SMPLH_FILE>`` and
    ``model_root/<SMPLX_FILE>`` by hand (SMPL-H: the AMASS "extended SMPL+H" male model.npz).
    """
    if all((model_root / name).is_file() for name in (SMPLH_FILE, SMPLX_FILE)):
        return 0
    files: list[HfFileInfo] = transports.hf_lfs_files(MODEL_REPO, [SMPLH_FILE, SMPLX_FILE], revision=MODEL_REVISION)
    return sum(transports.hf_fetch_verified(MODEL_REPO, file, local_dir=model_root, revision=MODEL_REVISION) for file in files)


@serde
@dataclass(frozen=True, slots=True)
class ManifestPath:
    """One file of a release part, as the official manifest lists it (unknown keys allowed: not our schema)."""

    source_path: str
    """Full ``s3://`` object path."""
    relative_path: str
    """Destination under the download root (``takes/<take_name>/frame_aligned_videos/cam01.mp4``)."""
    size: int | None = None
    """Bytes, when the manifest records them."""
    views: list[str] | None = None
    """``ego`` / ``exo`` for view-specific files."""


@serde
@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """Every file of one take (or capture) uid in one part."""

    uid: str
    """Take uid, capture uid, or a part-wide id (``metadata``)."""
    paths: list[ManifestPath]
    """Files under this uid."""


class Release:
    """Read access to the Ego-Exo4D release through one AWS profile (the licence keys)."""

    def __init__(self, profile: str) -> None:
        import s3fs

        self.profile: str = profile
        self.fs = s3fs.S3FileSystem(profile=profile)
        self.manifests: dict[str, dict[str, ManifestEntry]] = {}

    def manifest(self, part: Part) -> dict[str, ManifestEntry]:
        """One part's manifest keyed by uid, read once per process."""
        if part not in self.manifests:
            path: str = f"{RELEASE}/{part}/manifest.json"
            try:
                text: str = self.fs.cat_file(path).decode()
            except PermissionError as error:
                raise PermissionError(
                    f"{path}: access denied for AWS profile {self.profile!r}; put the Ego-Exo4D licence keys in that profile (https://ego4ddataset.com)"
                ) from error
            self.manifests[part] = {entry.uid: entry for entry in from_json(list[ManifestEntry], text)}
        return self.manifests[part]

    def fetch(self, file: ManifestPath, root: Path) -> bool:
        """Fetch one listed file to ``root/<relative_path>`` unless it already has the listed size; return whether bytes moved.

        The object lands under ``root/.dataforge-staging`` and is renamed into place only at the listed size, so a
        destination of that size was complete when it landed.
        """
        dest: Path = root / file.relative_path
        size: int = file.size if file.size is not None else int(self.fs.size(file.source_path))
        integrity: FileIntegrity = FileIntegrity(size, None, None)
        if transports.matches_file(dest, integrity):
            return False
        staged: Path = root / transports.STAGING_DIR / f"{file.relative_path}.{time.time_ns()}"
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fs.get_file(file.source_path, str(staged))
            transports.publish_verified(staged, dest, integrity)
        finally:
            staged.unlink(missing_ok=True)
        return True
