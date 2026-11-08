"""GSI's dataset API (``api.gsi.company``): partner sign-in, the edition manifest, and signed links to each episode's files.

Read from GSI's own CLI (``gsi`` 0.2.2, ``gsi_api/cli.py``, from ``https://api.gsi.company/install.sh``), which
dataforge does not install: a download is three calls. ``POST /v1/auth/partner-login`` trades the partner username and
password for a bearer token that lives 24 h; ``GET /v1/datasets/files`` pages through an edition, each item one episode
with its files' names, sizes, sha256 and a signed Cloudflare R2 link that lives an hour; ``POST /v1/datasets/manifest``
signs a link to the edition manifest. An R2 link is signed for GET only (HEAD answers 403), so ``transports.http_fetch``,
which probes with HEAD, cannot fetch it; ``fetch_signed`` streams one GET instead, as the CLI does.

The password comes from the environment only (``~/.secrets.env`` on our machines). A token and a signed link are
secrets too: none is printed, logged or written, and every record holding one keeps it out of ``repr``.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests
from serde import serde

from dataforge import transports, writing
from dataforge.records import decode

API_URL: str = "https://api.gsi.company"
"""GSI's API, the CLI's default ``--api-url``."""
USERNAME_ENV: str = "GSI_PARTNER_USERNAME"
"""Environment variable holding the partner username (GSI's README: ``gsi auth --partner <username>``)."""
PASSWORD_ENV: str = "GSI_PARTNER_PASSWORD"
"""Environment variable holding the partner password, which GSI hands over separately."""
PAGE_EPISODES: int = 25
"""Episodes per listing page, as the CLI asks: a page is listed just before its files are fetched, well inside the links' hour."""
TIMEOUT_S: float = 60.0
"""Bound on each API request and on each read of a file body."""


@serde
@dataclass(frozen=True, slots=True)
class PartnerTokens:
    """``POST /v1/auth/partner-login``'s answer, read partially (third-party format)."""

    access_token: str = field(repr=False)
    """Bearer token for every other call; a secret."""


@serde
@dataclass(frozen=True, slots=True)
class SignedFile:
    """One file of one episode in a listing page."""

    name: str
    """File name, ``hands.rrd`` or ``slam.rrd``."""
    size_bytes: int
    """Exact size."""
    sha256: str
    """Hex sha256 of the file."""
    url: str = field(repr=False)
    """Signed R2 link, valid for an hour; a secret."""

    @property
    def integrity(self) -> transports.FileIntegrity:
        """What the listing promises about the file."""
        return transports.FileIntegrity(self.size_bytes, "sha256", self.sha256)


@serde
@dataclass(frozen=True, slots=True)
class EpisodeFiles:
    """One episode of a listing page."""

    episode_id: str
    """GSI's episode id, ``ep_v1_<hex>``."""
    files: list[SignedFile]
    """Its files, each with a signed link."""


@serde
@dataclass(frozen=True, slots=True)
class FilesPage:
    """``GET /v1/datasets/files``'s answer, read partially."""

    items: list[EpisodeFiles]
    """The page's episodes, in the edition's order."""
    next_offset: int | None
    """Offset of the next page; ``None`` on the last."""


@serde
@dataclass(frozen=True, slots=True)
class SignedManifest:
    """``POST /v1/datasets/manifest``'s answer, read partially."""

    url: str = field(repr=False)
    """Signed link to the manifest JSON; a secret."""


@serde
@dataclass(frozen=True, slots=True)
class ManifestFile:
    """One recording of a manifest episode."""

    path: str
    """Edition-relative path, ``episodes/<episode_id>/rerun/<name>``."""
    sha256: str
    """Hex sha256."""
    size_bytes: int
    """Exact size."""


@serde
@dataclass(frozen=True, slots=True)
class ManifestEpisode:
    """One episode of the edition manifest."""

    episode_id: str
    """GSI's episode id."""
    caption: str
    """One-sentence description of the activity."""
    category: str
    """Task family derived from the caption, e.g. ``Food preparation``."""
    duration_s: float
    """Episode length in seconds."""
    views: int
    """Camera count; six in this dataset."""
    hands_rrd: ManifestFile
    """The six videos, the calibration and both hands' MANO tracks."""
    slam_rrd: ManifestFile
    """The rig trajectory."""


@serde
@dataclass(frozen=True, slots=True)
class Manifest:
    """GSI's portable edition manifest (``gsi.ots-portable-edition.v1``), read partially."""

    dataset: str
    """Dataset name, ``GSI Egocentric Six-View``."""
    edition: str
    """Edition name, ``10h``."""
    source_release: str
    """Upstream release the edition was cut from, ``2026-09-v2``."""
    rrd_contract: str
    """Layout contract of the shipped rrds, ``gsi.ots.customer-rrd.v2``."""
    privacy_status: str
    """Anonymization state, e.g. ``PRIVATE_INTERIM_EXISTING_BLUR``."""
    episodes: list[ManifestEpisode]
    """Every episode of the edition."""


def read_manifest(path: Path) -> Manifest:
    """Parse a manifest written by ``fetch_manifest``."""
    return decode(Manifest, path.read_text(), source=str(path))


@dataclass(frozen=True, slots=True)
class GsiSession:
    """A signed-in partner account."""

    api_url: str
    """API root, without a trailing slash."""
    token: str = field(repr=False)
    """Bearer token; a secret."""

    @classmethod
    def sign_in(cls, api_url: str) -> "GsiSession":
        """Trade the partner username and password from the environment for a token."""
        username: str | None = os.environ.get(USERNAME_ENV)
        password: str | None = os.environ.get(PASSWORD_ENV)
        if not username or not password:
            raise ValueError(f"GSI needs {USERNAME_ENV} and {PASSWORD_ENV} in the environment (keep them in ~/.secrets.env)")
        response: requests.Response = requests.post(f"{api_url}/v1/auth/partner-login", json={"username": username, "password": password}, timeout=TIMEOUT_S)
        if not response.ok:
            raise ValueError(f"GSI refused the sign-in of partner {username} (HTTP {response.status_code})")
        tokens: PartnerTokens = decode(PartnerTokens, response.text, source="GSI partner-login", redact="not a token answer")
        return cls(api_url, tokens.access_token)

    def call(self, method: Literal["GET", "POST"], path: str, *, params: dict[str, str | int] | None = None, body: dict[str, str] | None = None) -> str:
        """One authorized API call with query ``params`` or a JSON ``body``; the answer's text."""
        response: requests.Response = requests.request(
            method, f"{self.api_url}{path}", params=params, json=body, headers={"Authorization": f"Bearer {self.token}"}, timeout=TIMEOUT_S
        )
        if not response.ok:
            raise RuntimeError(f"GSI {method} {path} answered HTTP {response.status_code}")
        return response.text

    def files_page(self, edition: str, offset: int) -> FilesPage:
        """One listing page of ``edition`` starting at ``offset``, its links freshly signed."""
        text: str = self.call("GET", "/v1/datasets/files", params={"edition": edition, "offset": offset, "limit": PAGE_EPISODES})
        return decode(FilesPage, text, source=f"GSI files page at offset {offset}", redact="not a files page")

    def manifest_link(self, edition: str) -> str:
        """A signed link to ``edition``'s manifest."""
        text: str = self.call("POST", "/v1/datasets/manifest", body={"edition_id": edition})
        return decode(SignedManifest, text, source="GSI manifest link", redact="not a manifest link").url


def fetch_signed(url: str, dest: Path, *, label: str, expected: transports.FileIntegrity | None = None) -> None:
    """Stream one signed link to ``dest`` through a temp file, verified against ``expected`` before it takes the name.

    Raises:
        transports.IntegrityError: The bytes differ from ``expected``; ``dest`` is untouched.
        RuntimeError: The link answered an error (an expired link answers 403).
    """
    with writing.atomic_write(dest) as temp_path:
        with requests.get(url, stream=True, timeout=TIMEOUT_S) as response:
            if not response.ok:
                raise RuntimeError(f"{label}: the signed link answered HTTP {response.status_code}")
            with temp_path.open("wb") as sink:
                for block in response.iter_content(chunk_size=transports.CHUNK_BYTES):
                    sink.write(block)
        if expected is not None:
            transports.verify_file(temp_path, expected)


def fetch_manifest(session: GsiSession, edition: str, dest: Path) -> None:
    """Write ``edition``'s current manifest to ``dest``."""
    fetch_signed(session.manifest_link(edition), dest, label="GSI manifest")


def fetch_episodes(session: GsiSession, edition: str, wanted: set[str], destination: Callable[[str, str], Path]) -> transports.FetchReport:
    """Fetch every file of the ``wanted`` episodes to ``destination(episode_id, name)``, skipping files already there.

    The API has no per-episode query, so this pages through the listing until every wanted episode has gone by. A
    file whose link fails is fetched once more from a freshly signed page, which is how an expired link renews.

    Raises:
        ValueError: The listing ended before every wanted episode appeared.
    """
    report: transports.FetchReport = transports.FetchReport()
    remaining: set[str] = set(wanted)
    offset: int | None = 0
    while remaining and offset is not None:
        page: FilesPage = session.files_page(edition, offset)
        for index, episode in enumerate(page.items):
            if episode.episode_id not in remaining:
                continue
            for file in episode.files:
                dest: Path = destination(episode.episode_id, file.name)
                if transports.matches_file(dest, file.integrity):
                    report.count(False, file.size_bytes)
                    continue
                label: str = f"GSI {episode.episode_id}/{file.name}"
                try:
                    fetch_signed(file.url, dest, label=label, expected=file.integrity)
                except RuntimeError:
                    fresh: SignedFile = next(f for f in session.files_page(edition, offset).items[index].files if f.name == file.name)
                    fetch_signed(fresh.url, dest, label=label, expected=file.integrity)
                report.count(True, file.size_bytes)
            remaining.discard(episode.episode_id)
            print(f"  {episode.episode_id}: {len(wanted) - len(remaining)}/{len(wanted)} episodes")
        offset = page.next_offset
    if remaining:
        raise ValueError(f"GSI's listing of edition {edition} ended without {len(remaining)} wanted episode(s): {sorted(remaining)[:5]}")
    return report
