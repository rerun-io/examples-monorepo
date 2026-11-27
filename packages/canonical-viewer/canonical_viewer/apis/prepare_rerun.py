"""Prepare the patched checkout of the Rerun release tag that the canonical viewer's Cargo workspace builds against.

The workspace's ``[patch."https://github.com/rerun-io/rerun"]`` points at ``target/patch/rerun-<tag>/``: the tag's
source with every patch in ``RERUN_PATCHES`` applied, in order. The tree is rebuilt only when the commit or a patch
changes (a fingerprint file records both).
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]
RERUN_URL: str = "https://github.com/rerun-io/rerun"
RERUN_TAG: str = "0.44.0"
RERUN_COMMIT: str = "f346ba2728b9f55afdfd5afdb2b976fecb98dcdd"
"""The tag's commit, checked after a fetch."""
RERUN_PATCHES: tuple[str, ...] = ("patches/re_viewer-0.44.0-extension-hooks.patch",)
"""Patches relative to the package directory, applied in order with ``git apply -p1`` at the checkout's root."""
FINGERPRINT_FILE: str = ".prepared-sha256"


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


def rerun_archive(cargo_home: Path) -> bytes:
    """A tar of the Rerun tag's commit with entries under ``rerun-<tag>/``, from Cargo's clone when it has the commit."""
    with tempfile.TemporaryDirectory(prefix="rerun-fetch-") as temporary:
        # Rerun keeps test snapshots in LFS; git archive would download them through the smudge filter.
        git_env: dict[str, str] = dict(os.environ, GIT_CEILING_DIRECTORIES=temporary, GIT_LFS_SKIP_SMUDGE="1")
        repository: Path | None = next(
            (
                database
                for database in sorted((cargo_home / "git/db").glob("rerun-*"))
                if subprocess.run(["git", f"--git-dir={database}", "cat-file", "-e", f"{RERUN_COMMIT}^{{commit}}"], capture_output=True).returncode == 0
            ),
            None,
        )
        if repository is None:
            repository = Path(temporary) / "repository.git"
            subprocess.run(["git", "init", "-q", "--bare", str(repository)], env=git_env, check=True)
            subprocess.run(["git", f"--git-dir={repository}", "fetch", "-q", "--depth", "1", RERUN_URL, f"refs/tags/{RERUN_TAG}"], env=git_env, check=True)
            fetched: str = subprocess.run(
                ["git", f"--git-dir={repository}", "rev-parse", "FETCH_HEAD^{commit}"], env=git_env, check=True, capture_output=True, text=True
            ).stdout.strip()
            if fetched != RERUN_COMMIT:
                raise ValueError(f"Rerun tag {RERUN_TAG} is {fetched}, expected {RERUN_COMMIT}")
        return subprocess.run(
            ["git", f"--git-dir={repository}", "archive", "--format=tar", f"--prefix=rerun-{RERUN_TAG}/", RERUN_COMMIT],
            env=git_env,
            check=True,
            capture_output=True,
        ).stdout


@dataclass
class PrepareRerunConfig:
    """Prepare target/patch/rerun-<tag>/ for the canonical viewer's Cargo workspace."""

    cargo_home: Path = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    """Cargo home whose git clones may already hold the tag's commit."""
    target_dir: Path = PACKAGE_DIR / "target"
    """Directory that holds patch/rerun-<tag>/ (the path the workspace's [patch] section names)."""


def main(config: PrepareRerunConfig) -> None:
    """Prepare the patched Rerun checkout, unless an identical one is already there."""
    destination: Path = config.target_dir / "patch" / f"rerun-{RERUN_TAG}"
    built: bool = prepare_tree(
        destination,
        lambda: rerun_archive(config.cargo_home),
        source_id=RERUN_COMMIT,
        patches=[PACKAGE_DIR / patch for patch in RERUN_PATCHES],
    )
    print(f"{destination}: {'prepared' if built else 'unchanged'}")
