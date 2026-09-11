"""Materialize checksummed Cargo archives and local patches before Cargo runs."""

import fcntl
import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PatchedCrate:
    """Pinned archive and its crate-relative patch."""

    name: str
    """Crates.io package name."""
    version: str
    """Exact published version."""
    sha256: str
    """Expected archive digest."""
    patch: str
    """Patch path relative to the SLAM package."""


PATCHED_CRATES: tuple[PatchedCrate, ...] = (
    PatchedCrate(
        'cubecl-common',
        '0.11.0-pre.3',
        '4d40f7451ab096a127c58b03d76b6d30f4fc757ac13d8e2f17ed5bb20ffda4bc',
        'patches/cubecl-common-0.11.0-pre.3-channel-park.patch',
    ),
)


# src/device/handle/channel.rs at CubeCL fork commit 93d463c9.
CHANNEL_SHA256: str = '75e4b83cb12ad4bfa07b2e72c8e7fafdb363c00f98682174f03dafc8f5e849d7'


@dataclass(frozen=True, slots=True)
class Config:
    """Prepare the dependencies of this source checkout."""

    package_dir: Path = Path(__file__).resolve().parents[2]
    """SLAM package containing Cargo.toml and patches/."""


def prepare(crate: PatchedCrate, package_dir: Path, cargo_home: Path) -> None:
    """Verify, extract and patch one crate; retain a matching prepared tree."""
    package_dir = package_dir.resolve()
    stem: str = f'{crate.name}-{crate.version}'
    destination: Path = package_dir / 'target/patch' / stem
    # POSIX-only: the repository has Linux and macOS lanes, no Windows lane.
    # Keep this inode beside the destination; never unlink a lock with waiters.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f'.{stem}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        patch: Path = package_dir / crate.patch
        # An opaque fingerprint avoids parsing a marker schema: compare exact bytes.
        fingerprint: str = f'{crate.sha256}\n{hashlib.sha256(patch.read_bytes()).hexdigest()}\n'
        marker: Path = destination / '.prepared-sha256'
        if marker.is_file() and marker.read_text() == fingerprint and (destination / 'Cargo.toml').is_file():
            print(f'{stem}: unchanged')
            return
        cached: Path | None = next((cargo_home / 'registry/cache').glob(f'*/{stem}.crate'), None)
        archive: bytes
        if cached is not None:
            archive = cached.read_bytes()
        else:
            with urllib.request.urlopen(f'https://static.crates.io/crates/{crate.name}/{stem}.crate', timeout=60) as response:
                archive = response.read()
        if hashlib.sha256(archive).hexdigest() != crate.sha256:
            raise ValueError(f'{stem}: archive SHA256 mismatch')
        with tempfile.TemporaryDirectory(prefix=f'.{stem}-', dir=destination.parent) as temporary:
            staging: Path = Path(temporary)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(staging, filter='data')
            extracted: Path = staging / stem
            # Do not discover the enclosing worktree: git otherwise skips paths
            # beneath its ignored target directory while returning success.
            subprocess.run(
                ['git', 'apply', '-p1', f'--directory={stem}', str(patch)],
                cwd=staging,
                env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(staging.parent)),
                check=True,
            )
            (extracted / '.prepared-sha256').write_text(fingerprint)
            if destination.exists():
                destination.rename(staging / "previous")
            extracted.rename(destination)
        print(f'{stem}: prepared ({"cache" if cached is not None else "download"})')


def main(config: Config) -> None:
    """Prepare every pinned patch using Cargo's configured archive cache."""
    cargo_home: Path = Path(os.environ.get('CARGO_HOME', str(Path.home() / '.cargo')))
    for crate in PATCHED_CRATES:
        prepare(crate, config.package_dir, cargo_home)
