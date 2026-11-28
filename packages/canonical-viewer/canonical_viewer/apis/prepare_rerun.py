"""Prepare the patched source trees that the canonical viewer's Cargo workspace builds against.

The workspace's ``[patch."https://github.com/rerun-io/rerun"]`` points at ``target/patch/rerun-<tag>/``: the tag's
source with every patch in ``RERUN_PATCHES`` applied, in order. Its ``[patch.crates-io]`` points at
``target/patch/<crate>-<version>/`` for each crate in ``PATCHED_CRATES``: the verified crates.io archive with its patch.
A tree is rebuilt only when its source or a patch changes (a fingerprint file records both).
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from canonical_viewer import PACKAGE_DIR

RERUN_URL: str = "https://github.com/rerun-io/rerun"
RERUN_TAG: str = "0.44.0"
RERUN_COMMIT: str = "f346ba2728b9f55afdfd5afdb2b976fecb98dcdd"
"""The tag's commit, checked after a fetch."""
RERUN_PATCHES: tuple[str, ...] = (
    "patches/re_viewer-0.44.0-extension-hooks.patch",
    # From ai-demos .github/rerun-source-build/ (main acaa65f), which builds the same fixes into its rerun-sdk wheels:
    # native AV1 decoding on Linux arm64 (rerun-io/rerun#7755; re_rav1d 0.1.4 from a pinned git revision) and a stalled
    # live-proxy reader that no longer blocks the logger and every other viewer.
    "patches/rerun-0.44.0-av1-arm64.patch",
    "patches/rerun-0.44.0-proxy-slow-reader.patch",
)
"""Patches relative to the package directory, applied in order with ``git apply -p1`` at the checkout's root."""
FINGERPRINT_FILE: str = ".prepared-sha256"


@dataclass(frozen=True, slots=True)
class PatchedCrate:
    """A pinned crates.io archive and the patch applied to it."""

    name: str
    version: str
    sha256: str
    """Archive digest, as in Cargo.lock."""
    patch: str
    """Patch path relative to the package directory, applied at the crate's root."""


@dataclass(frozen=True, slots=True)
class PatchedGitCrate:
    """A crate pinned to a git commit and the patch applied to it; prepared as ``<name>-<commit[:8]>``."""

    name: str
    url: str
    commit: str
    patch: str
    """Patch path relative to the package directory, applied at the crate's root."""

    @property
    def stem(self) -> str:
        return f"{self.name}-{self.commit[:8]}"


PATCHED_GIT_CRATES: tuple[PatchedGitCrate, ...] = (
    # The AV1 patch's re_rav1d (rerun-io/rerun#7755), made to build for wasm32 so the web viewer decodes depth video with
    # the same decoder as native: libc has none of the C types and errno values rav1d names there.
    PatchedGitCrate(
        "re_rav1d",
        "https://github.com/pablo-rerun/re_rav1d.git",
        "dd03865f74aee68b90269ef3b4e33fbb494ffca3",
        "patches/re_rav1d-0.1.4-wasm-libc.patch",
    ),
)

PATCHED_CRATES: tuple[PatchedCrate, ...] = (
    # Maps Features::SUBGROUP to WebGPU "subgroups" (the web backend only), so the splat kernels can use subgroups in browsers.
    PatchedCrate("wgpu", "30.0.0", "6d8f4bd44d92da5270f03409dba9f952dab24f128e05d6a554926101d1bf9114", "patches/wgpu-30.0.0-webgpu-subgroups.patch"),
)


def prepare_tree(destination: Path, archive: Callable[[], bytes], *, source_id: str, patches: list[Path]) -> bool:
    """Extract a tar archive whose entries start with ``destination.name``, apply the patches and swap the tree in.

    Args:
        destination: The prepared tree; the archive's top-level directory has its name.
        archive: Fetches the tar archive; only called when the tree is missing or stale.
        source_id: Commit or digest that, with the patches' digests, fingerprints the tree.
        patches: Patch files applied in order at the tree's root.

    Returns:
        True when the tree was (re)built, False when an identical tree was already there.
    """
    fingerprint: str = "\n".join([source_id, *(hashlib.sha256(patch.read_bytes()).hexdigest() for patch in patches)]) + "\n"
    marker: Path = destination / FINGERPRINT_FILE
    if marker.is_file() and marker.read_text() == fingerprint:
        return False
    data: bytes = archive()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging: Path = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(staging, filter="data")
        tree: Path = staging / destination.name
        # The ceiling keeps git from discovering an enclosing worktree, which would skip paths under its ignored target/.
        git_env: dict[str, str] = dict(os.environ, GIT_CEILING_DIRECTORIES=str(staging))
        for patch in patches:
            subprocess.run(["git", "apply", "-p1", str(patch)], cwd=tree, env=git_env, check=True)
        (tree / FINGERPRINT_FILE).write_text(fingerprint)
        if destination.exists():
            destination.rename(staging / "previous")
        tree.rename(destination)
    return True


def git_archive(url: str, commit: str, ref: str, prefix: str, cargo_home: Path, cargo_db: str) -> bytes:
    """A tar of `commit` with entries under ``<prefix>/``, from Cargo's clone (`cargo_db`, a glob under git/db) when it
    has the commit, else from a shallow fetch of `ref` from `url`, checked to be `commit`."""
    with tempfile.TemporaryDirectory(prefix="git-fetch-") as temporary:
        # Rerun keeps test snapshots in LFS; git archive would download them through the smudge filter.
        git_env: dict[str, str] = dict(os.environ, GIT_CEILING_DIRECTORIES=temporary, GIT_LFS_SKIP_SMUDGE="1")
        repository: Path | None = next(
            (
                database
                for database in sorted((cargo_home / "git/db").glob(cargo_db))
                if subprocess.run(["git", f"--git-dir={database}", "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode == 0
            ),
            None,
        )
        if repository is None:
            repository = Path(temporary) / "repository.git"
            subprocess.run(["git", "init", "-q", "--bare", str(repository)], env=git_env, check=True)
            subprocess.run(["git", f"--git-dir={repository}", "fetch", "-q", "--depth", "1", url, ref], env=git_env, check=True)
            fetched: str = subprocess.run(
                ["git", f"--git-dir={repository}", "rev-parse", "FETCH_HEAD^{commit}"], env=git_env, check=True, capture_output=True, text=True
            ).stdout.strip()
            if fetched != commit:
                raise ValueError(f"{url} {ref} is {fetched}, expected {commit}")
        return subprocess.run(
            ["git", f"--git-dir={repository}", "archive", "--format=tar", f"--prefix={prefix}/", commit],
            env=git_env,
            check=True,
            capture_output=True,
        ).stdout


def rerun_archive(cargo_home: Path) -> bytes:
    """A tar of the Rerun tag's commit with entries under ``rerun-<tag>/``."""
    return git_archive(RERUN_URL, RERUN_COMMIT, f"refs/tags/{RERUN_TAG}", f"rerun-{RERUN_TAG}", cargo_home, "rerun-*")


def crate_archive(crate: PatchedCrate, cargo_home: Path) -> bytes:
    """The crate's .crate archive (entries under ``<name>-<version>/``), from Cargo's cache or crates.io, digest checked."""
    stem: str = f"{crate.name}-{crate.version}"
    cached: Path | None = next((cargo_home / "registry/cache").glob(f"*/{stem}.crate"), None)
    if cached is not None:
        archive: bytes = cached.read_bytes()
    else:
        with urllib.request.urlopen(f"https://static.crates.io/crates/{crate.name}/{stem}.crate", timeout=60) as response:
            archive = response.read()
    if hashlib.sha256(archive).hexdigest() != crate.sha256:
        raise ValueError(f"{stem}: archive SHA256 mismatch")
    return archive


@dataclass
class PrepareRerunConfig:
    """Prepare target/patch/rerun-<tag>/ and the patched crates for the canonical viewer's Cargo workspace."""

    cargo_home: Path = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    """Cargo home whose git clones may already hold the tag's commit."""
    target_dir: Path = PACKAGE_DIR / "target"
    """Directory that holds patch/ (the paths the workspace's [patch] sections name)."""


def main(config: PrepareRerunConfig) -> None:
    """Prepare the patched Rerun checkout and each patched crate, unless an identical tree is already there."""
    destination: Path = config.target_dir / "patch" / f"rerun-{RERUN_TAG}"
    built: bool = prepare_tree(
        destination,
        lambda: rerun_archive(config.cargo_home),
        source_id=RERUN_COMMIT,
        patches=[PACKAGE_DIR / patch for patch in RERUN_PATCHES],
    )
    print(f"{destination}: {'prepared' if built else 'unchanged'}")
    for crate in PATCHED_CRATES:
        destination = config.target_dir / "patch" / f"{crate.name}-{crate.version}"
        built = prepare_tree(
            destination,
            partial(crate_archive, crate, config.cargo_home),
            source_id=crate.sha256,
            patches=[PACKAGE_DIR / crate.patch],
        )
        print(f"{destination}: {'prepared' if built else 'unchanged'}")
    for crate in PATCHED_GIT_CRATES:
        destination = config.target_dir / "patch" / crate.stem
        built = prepare_tree(
            destination,
            partial(git_archive, crate.url, crate.commit, crate.commit, crate.stem, config.cargo_home, f"{crate.name}-*"),
            source_id=crate.commit,
            patches=[PACKAGE_DIR / crate.patch],
        )
        print(f"{destination}: {'prepared' if built else 'unchanged'}")
