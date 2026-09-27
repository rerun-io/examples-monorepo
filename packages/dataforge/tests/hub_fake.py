"""One fake Hugging Face Hub every Hub-backed download test runs against.

Not a test module. ``HubStore`` holds dataset repos in memory and answers the
four huggingface_hub entry points dataforge calls — ``HfApi`` (tree listing,
path info, auth check, repo info), ``HfFileSystem.open``, ``hf_hub_download``
and ``snapshot_download`` — with real ``RepoFile`` entries (size, git blob sha1,
LFS sha256), so the production listing, verification and publish code runs
unchanged and only the network is faked. ``install`` swaps those names in the
modules a test names. What a dataset's fixture holds, and how a test breaks a
transfer, stays with the dataset's test module.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, NamedTuple

import pytest
from huggingface_hub import HfApi, HfFileSystem, RepoFile


class HubCall(NamedTuple):
    """One request the store answered."""

    kind: Literal["list", "info", "open", "fetch", "snapshot"]
    """Tree listing, path-info lookup, range-read open, single-file download, or pattern snapshot."""
    repo_id: str
    """Repo the request named."""
    revision: str | None
    """Revision the request asked for."""
    paths: tuple[str, ...]
    """Listed folder, looked-up paths, the opened or downloaded file, or the snapshot's ``allow_patterns``."""
    force: bool = False
    """``force_download`` of a single-file download."""


class HubStore:
    """Dataset repos in memory, served at pinned revisions; every request is logged in ``calls``."""

    def __init__(self, pins: Mapping[str, str] | None = None) -> None:
        """``pins`` maps a repo to the only revision a request may ask it for; unpinned repos accept any."""
        self.pins: dict[str, str] = dict(pins or {})
        self.files: dict[tuple[str, str], bytes] = {}
        """``(repo, path)`` → the file's bytes, in the order the repo lists them."""
        self.lfs: set[tuple[str, str]] = set()
        """Files stored in LFS: their listing carries a sha256."""
        self.served: dict[tuple[str, str], bytes] = {}
        """Bytes a download writes instead of ``files``; a test puts a bad transfer here."""
        self.sizes: dict[tuple[str, str], int] = {}
        """Size a listing advertises instead of the real one."""
        self.refused: dict[str, Exception] = {}
        """Repo → the error every request on it raises (gated, denied)."""
        self.calls: list[HubCall] = []
        store: HubStore = self

        class Api(HfApi):
            """``HfApi`` over this store; a subclass because the datasets annotate their ``api: HfApi`` parameters."""

            def list_repo_tree(self, repo_id: str, **kwargs):  # type: ignore[override]
                return store.list_repo_tree(repo_id, **kwargs)

            def get_paths_info(self, repo_id: str, paths, **kwargs):  # type: ignore[override]
                return store.get_paths_info(repo_id, paths, **kwargs)

            def auth_check(self, repo_id: str, **kwargs):  # type: ignore[override]
                return store.auth_check(repo_id, **kwargs)

            def repo_info(self, repo_id: str, **kwargs):  # type: ignore[override]
                return store.repo_info(repo_id, **kwargs)

        class FileSystem(HfFileSystem):
            """``HfFileSystem.open`` over this store."""

            def open(self, path, mode="rb", **kwargs):  # type: ignore[override]
                return store.open(path)

        self.api: type[HfApi] = Api
        self.filesystem: type[HfFileSystem] = FileSystem

    def add(self, repo_id: str, path: str, data: bytes, *, lfs: bool = False) -> None:
        """Put one file into a repo."""
        self.files[repo_id, path] = data
        if lfs:
            self.lfs.add((repo_id, path))

    def add_tree(self, repo_id: str, root: Path, *, lfs: tuple[str, ...] = ()) -> None:
        """Put every file under ``root`` into a repo at its ``root``-relative path; names ending in ``lfs`` go to LFS."""
        for path in sorted(root.rglob("*")):
            if path.is_file():
                relative: str = path.relative_to(root).as_posix()
                self.add(repo_id, relative, path.read_bytes(), lfs=relative.endswith(lfs))

    @property
    def fetched(self) -> list[str]:
        """Every file a single-file download wrote, in order."""
        return [call.paths[0] for call in self.calls if call.kind == "fetch"]

    @property
    def forced(self) -> list[str]:
        """The downloads that passed ``force_download=True``."""
        return [call.paths[0] for call in self.calls if call.kind == "fetch" and call.force]

    @property
    def listed(self) -> list[str]:
        """The repo of every tree listing, in order."""
        return [call.repo_id for call in self.calls if call.kind == "list"]

    def entry(self, repo_id: str, path: str) -> RepoFile:
        """The listing entry the Hub returns for one file."""
        data: bytes = self.files[repo_id, path]
        size: int = self.sizes.get((repo_id, path), len(data))
        blob: str = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        lfs = {"size": size, "oid": hashlib.sha256(data).hexdigest(), "pointerSize": 133} if (repo_id, path) in self.lfs else None
        return RepoFile(path=path, size=size, oid=blob, lfs=lfs)

    def request(self, kind: Literal["list", "info", "open", "fetch", "snapshot"], repo_id: str, revision: str | None, paths: tuple[str, ...], force: bool = False) -> None:
        """Log one request, then refuse it when its repo is refused or its revision is not the pinned one."""
        self.calls.append(HubCall(kind, repo_id, revision, paths, force))
        if repo_id in self.refused:
            raise self.refused[repo_id]
        if repo_id in self.pins:
            assert revision == self.pins[repo_id], f"{repo_id} asked at {revision}, pinned {self.pins[repo_id]}"

    def list_repo_tree(
        self, repo_id: str, *, repo_type: str, revision: str | None = None, path_in_repo: str | None = None, recursive: bool = False
    ) -> list[RepoFile]:
        """Files under ``path_in_repo`` (the repo root when ``None``); only its direct children unless ``recursive``."""
        assert repo_type == "dataset"
        self.request("list", repo_id, revision, (path_in_repo or "",))
        prefix: str = f"{path_in_repo.rstrip('/')}/" if path_in_repo else ""
        found: list[str] = [path for repo, path in self.files if repo == repo_id and path.startswith(prefix)]
        return [self.entry(repo_id, path) for path in found if recursive or "/" not in path.removeprefix(prefix)]

    def get_paths_info(self, repo_id: str, paths: Sequence[str], *, repo_type: str, revision: str | None = None) -> list[RepoFile]:
        """Entries of the named files; an absent path is left out, as the Hub does."""
        assert repo_type == "dataset"
        self.request("info", repo_id, revision, tuple(paths))
        return [self.entry(repo_id, path) for path in paths if (repo_id, path) in self.files]

    def auth_check(self, repo_id: str, *, repo_type: str) -> None:
        """Raise the repo's refusal, if any."""
        assert repo_type == "dataset"
        if repo_id in self.refused:
            raise self.refused[repo_id]

    def repo_info(self, repo_id: str, *, repo_type: str, revision: str | None = None) -> SimpleNamespace:
        """The pinned revision as the resolved commit sha."""
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.pins.get(repo_id, revision))

    def open(self, path: str) -> io.BytesIO:
        """``datasets/<repo>@<revision>/<path>`` as a seekable stream, the way ``HfFileSystem`` range-reads it."""
        repo_id, _, rest = path.removeprefix("datasets/").partition("@")
        revision, _, file_path = rest.partition("/")
        self.request("open", repo_id, revision, (file_path,))
        return io.BytesIO(self.files[repo_id, file_path])

    def hf_hub_download(
        self, repo_id: str, filename: str, *, repo_type: str, revision: str | None = None, local_dir: str, force_download: bool = False
    ) -> str:
        """Write one file under ``local_dir`` (``served`` bytes when a test set them) and return its path."""
        assert repo_type == "dataset"
        self.request("fetch", repo_id, revision, (filename,), force_download)
        target: Path = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.served.get((repo_id, filename), self.files[repo_id, filename]))
        return str(target)

    def snapshot_download(self, repo_id: str, *, repo_type: str, allow_patterns: list[str], local_dir: str, revision: str | None = None) -> str:
        """Write every file matching ``allow_patterns`` under ``local_dir``."""
        assert repo_type == "dataset"
        self.request("snapshot", repo_id, revision, tuple(allow_patterns))
        for repo, path in self.files:
            if repo == repo_id and any(fnmatch.fnmatch(path, pattern) for pattern in allow_patterns):
                target: Path = Path(local_dir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.served.get((repo, path), self.files[repo, path]))
        return local_dir

    def install(self, monkeypatch: pytest.MonkeyPatch, *modules: ModuleType) -> None:
        """Point every huggingface_hub name each of ``modules`` imported at this store."""
        fakes: dict[str, object] = {
            "HfApi": self.api,
            "HfFileSystem": self.filesystem,
            "hf_hub_download": self.hf_hub_download,
            "snapshot_download": self.snapshot_download,
        }
        for module in modules:
            for name, fake in fakes.items():
                if hasattr(module, name):
                    monkeypatch.setattr(module, name, fake)
