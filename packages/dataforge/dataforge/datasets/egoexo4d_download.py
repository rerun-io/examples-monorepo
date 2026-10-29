"""Ego-Exo4D raw tree: HM fits and body models from pinned Hugging Face commits, take files from the Ego-Exo4D S3 release.

The release needs the Ego-Exo4D licence keys (https://ego4ddataset.com), read as an AWS profile. Its manifest
(``<release>/<part>/manifest.json``, the file the official ``egoexo`` CLI reads) lists every file of a part per
take or capture uid with its size, so dataforge fetches exactly the files a take's conversion reads and nothing
else, verified by size, into the layout the CLI writes (``<root>/<relative_path>``).
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from aiohttp import ClientError as AioClientError
from botocore.exceptions import ClientError, HTTPClientError, IncompleteReadError
from botocore.exceptions import ConnectionError as S3ConnectionError
from fsspec.asyn import sync
from serde import serde

from dataforge import transports
from dataforge.datasets.egoexo4d_body import HM_FILE, HM_REPO, HM_REVISION
from dataforge.records import decode
from dataforge.transports import FileIntegrity, HfFileInfo

RANGE_BYTES: int = 32 * 1024 * 1024
"""Bytes per retryable S3 range (at most sixteen streams across takes)."""

NO_PROGRESS_S: float = 60.0
"""Abort a GET or body read that produces no application bytes for this long."""
RANGE_DEADLINE_S: float = 180.0
"""Hard deadline per range attempt, including a stream that only trickles bytes."""
RANGE_ATTEMPTS: int = 3
RETRY_DELAY_S: float = 1.0


@runtime_checkable
class S3Body(Protocol):
    """The streaming part of an S3 GetObject response."""

    async def read(self, amount: int) -> bytes: ...

    def close(self) -> None: ...


RELEASE: str = "s3://ego4d-consortium-sharing/egoexo-public/v2"
"""The public Ego-Exo4D release the official ``egoexo`` CLI downloads (``--release v2``)."""
HM_DIR: str = "hm"
"""Raw-root subdirectory holding ``<take_name>/<HM_FILE>``."""
Part: TypeAlias = Literal["metadata", "takes", "take_trajectory", "captures", "take_vrs_noimagestream"]
"""Ego-Exo4D release parts this dataset reads."""


def hm_fits() -> dict[str, HfFileInfo]:
    """Every take the pinned HM release ships a fit for, keyed by take name; one recursive listing call."""
    return {
        file.path.split("/")[0]: file
        for file in transports.hf_list_files(HM_REPO, HM_REVISION)
        if file.path.endswith(f"/{HM_FILE}") and file.path.count("/") == 1
    }


def fit_path(root: Path, take: str) -> Path:
    """Where download() puts one take's fit."""
    return root / HM_DIR / take / HM_FILE


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


@serde
@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """Every file of one take (or capture) uid in one part."""

    uid: str
    """Take uid, capture uid, or a part-wide id (``metadata``)."""
    paths: list[ManifestPath]
    """Files under this uid."""


class Release:
    """Read access to the Ego-Exo4D release through one AWS profile (the licence keys); safe to share across threads."""

    def __init__(self, profile: str) -> None:
        import s3fs  # simplecv declares it; only the release needs it

        self.profile: str = profile
        self.fs = s3fs.S3FileSystem(
            profile=profile,
            config_kwargs={"max_pool_connections": 16, "connect_timeout": 5, "read_timeout": 15, "retries": {"total_max_attempts": 1}},
        )
        self.fs.retries = 1  # Our range loop owns retries; do not multiply SDK and s3fs attempts.
        self.stream_slots = threading.BoundedSemaphore(16)
        self.manifests: dict[str, dict[str, ManifestEntry]] = {}
        self.lock = threading.Lock()  # threading.Lock is a factory, not a type beartype can check

    def manifest(self, part: Part) -> dict[str, ManifestEntry]:
        """One part's manifest keyed by uid, read once per process (convert and its prefetch thread share it)."""
        with self.lock:
            if part not in self.manifests:
                path: str = f"{RELEASE}/{part}/manifest.json"
                try:
                    with TemporaryDirectory(prefix="egoexo4d-manifest-") as directory:
                        root: Path = Path(directory)
                        self.fetch(ManifestPath(path, f"{part}/manifest.json"), root)
                        text: str = (root / part / "manifest.json").read_text()
                except PermissionError as error:
                    raise PermissionError(
                        f"{path}: access denied for AWS profile {self.profile!r}; put the Ego-Exo4D licence keys in that profile (https://ego4ddataset.com)"
                    ) from error
                self.manifests[part] = {entry.uid: entry for entry in decode(list[ManifestEntry], text, source=path)}
            return self.manifests[part]

    async def write_range(self, file: ManifestPath, staged: Path, start: int, end: int) -> None:
        """Stream a verified range with cancellable progress and attempt deadlines, closing every response."""
        parts: list[str] = file.source_path.removeprefix("s3://").split("/", 1)
        bucket: str = parts[0]
        key: str = parts[1]
        for attempt in range(1, RANGE_ATTEMPTS + 1):
            offset: int = start
            try:
                async with asyncio.timeout(RANGE_DEADLINE_S):
                    async with asyncio.timeout(NO_PROGRESS_S):
                        await self.fs.set_session()
                        client = self.fs.s3
                        assert client is not None
                        response = await client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end - 1}")
                    body: S3Body = response["Body"]
                    try:
                        with staged.open("r+b") as output:
                            output.seek(start)
                            while offset < end:
                                chunk: bytes = await asyncio.wait_for(body.read(min(1024 * 1024, end - offset)), timeout=NO_PROGRESS_S)
                                if not chunk or len(chunk) > end - offset:
                                    raise ValueError(f"{file.source_path}: range {start}:{end} received {offset - start} bytes, expected {end - start}")
                                output.write(chunk)
                                offset += len(chunk)
                    finally:
                        body.close()
                return
            except ClientError as error:
                status: int = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
                if status == 403:
                    raise PermissionError(f"{file.source_path}: access denied") from error
                if status < 500 and status not in (408, 429):
                    raise
                failure: str = f"HTTP {status}"
                if attempt == RANGE_ATTEMPTS:
                    print(f"S3 {file.relative_path} range {start}:{end}: attempt {attempt}/{RANGE_ATTEMPTS} failed ({failure}); giving up", flush=True)
                    raise
            except (TimeoutError, ConnectionError, S3ConnectionError, HTTPClientError, IncompleteReadError, AioClientError) as error:
                failure = type(error).__name__
                if attempt == RANGE_ATTEMPTS:
                    print(f"S3 {file.relative_path} range {start}:{end}: attempt {attempt}/{RANGE_ATTEMPTS} failed ({failure}); giving up", flush=True)
                    raise
            print(
                f"S3 {file.relative_path} range {start}:{end}: attempt {attempt}/{RANGE_ATTEMPTS} failed ({failure}), "
                f"{offset - start} bytes received; retrying (idle limit {NO_PROGRESS_S:g}s, attempt limit {RANGE_DEADLINE_S:g}s)", flush=True,
            )
            await asyncio.sleep(RETRY_DELAY_S * attempt)

    def fetch(self, file: ManifestPath, root: Path) -> bool:
        """Fetch one listed file to ``root/<relative_path>`` unless it already has the listed size; return whether bytes moved.

        The object lands under ``root/.dataforge-staging`` and is renamed into place only at the listed size, so a
        destination of that size was complete when it landed. The staging name is unique per call because convert and the
        next take's prefetch can fetch the same capture's ``timesync.csv`` at once.
        """
        dest: Path = root / file.relative_path
        size: int | None = file.size
        if size is None:
            remote_size = sync(self.fs.loop, self.fs._size, file.source_path, timeout=NO_PROGRESS_S)
            assert isinstance(remote_size, int)
            size = remote_size
        integrity: FileIntegrity = FileIntegrity(size, None, None)
        if transports.matches_file(dest, integrity):
            return False
        staged: Path = root / transports.STAGING_DIR / f"{file.relative_path}.{time.time_ns()}"
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"S3 fetch {file.relative_path}: {size} bytes", flush=True)
            with staged.open("wb") as output:
                output.truncate(size)

            def fetch_range(start: int) -> None:
                end: int = min(start + RANGE_BYTES, size)
                with self.stream_slots:
                    sync(
                        self.fs.loop, self.write_range, file, staged, start, end,
                        timeout=RANGE_ATTEMPTS * RANGE_DEADLINE_S + RETRY_DELAY_S * sum(range(1, RANGE_ATTEMPTS)) + 5.0,
                    )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(fetch_range, range(0, size, RANGE_BYTES)))
            transports.publish_verified(staged, dest, integrity)
            print(f"S3 complete {file.relative_path}: {size} bytes", flush=True)
        finally:
            staged.unlink(missing_ok=True)
        return True
