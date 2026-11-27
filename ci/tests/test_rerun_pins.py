"""Rerun pin guards: the Rust crates, their egui patches and the patched Rerun checkouts follow the conda rerun-sdk pin.

Rerun 0.44.0 is not on crates.io, so every Rust package takes the `re_*` crates from the rerun-io/rerun release tag
and patches the egui family, egui_crates and re_mp4 to the revisions in that tag's Cargo.lock. A bump that misses one
manifest builds a second egui or a viewer a release behind, and nothing else in CI compiles Rust to notice.
"""

import re
import subprocess
from pathlib import Path
from typing import Any

import tomllib

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
RERUN_GIT: str = "https://github.com/rerun-io/rerun"
PATCHED_GIT: tuple[str, ...] = (
    "https://github.com/emilk/egui.git",
    "https://github.com/rerun-io/egui_crates.git",
    "https://github.com/rerun-io/re_mp4.git",
)
CANONICAL_PREPARE: Path = REPO_ROOT / "packages/canonical-viewer/canonical_viewer/apis/prepare_rerun.py"
# A staged copy of CANONICAL_PREPARE until gsplat's web build moves onto the canonical viewer's web crate.
WEB_VIEWER_BUILD: Path = REPO_ROOT / "packages/gsplat-rust-renderer/gsplat_rust_renderer/apis/build_web_viewer.py"


def _cargo_manifests() -> list[Path]:
    """Every tracked Cargo.toml under packages/ (untracked target/ trees hold upstream checkouts)."""
    listed: str = subprocess.run(
        ["git", "ls-files", "-z", "--", "packages/*Cargo.toml"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    return [REPO_ROOT / name for name in listed.split("\0") if name.endswith("/Cargo.toml")]


def _git_sources(table: dict[str, Any], where: str) -> list[tuple[str, dict[str, Any]]]:
    """(location, entry) for every inline table with a `git` key, at any depth."""
    found: list[tuple[str, dict[str, Any]]] = []
    for key, value in table.items():
        if isinstance(value, dict):
            found += [(f"{where}.{key}", value)] if "git" in value else _git_sources(value, f"{where}.{key}")
    return found


def _conda_rerun_version() -> str:
    pin: dict[str, Any] = tomllib.loads((REPO_ROOT / "pixi.toml").read_text())["feature"]["common"]["dependencies"]["rerun-sdk"]
    match: re.Match[str] | None = re.fullmatch(r"==(\d+\.\d+\.\d+)", pin["version"])
    assert match is not None, f"[feature.common.dependencies] rerun-sdk: expected an exact ==X.Y.Z pin, got {pin['version']!r}"
    return match.group(1)


def test_rust_rerun_crates_use_the_conda_pin_tag() -> None:
    version: str = _conda_rerun_version()
    tags: dict[str, set[str]] = {}
    for manifest in _cargo_manifests():
        relative: str = str(manifest.relative_to(REPO_ROOT))
        for where, entry in _git_sources(tomllib.loads(manifest.read_text()), relative):
            if entry["git"].removesuffix(".git") == RERUN_GIT:
                tags.setdefault(str(entry.get("tag")), set()).add(where)
    assert tags, "no Cargo.toml takes crates from the rerun-io/rerun git tag"
    assert set(tags) == {version}, f"rerun-io/rerun git deps must all use tag {version!r} (the conda rerun-sdk pin): {tags}"


def test_patched_egui_family_uses_one_rev_per_repository() -> None:
    revs: dict[str, dict[str, set[str]]] = {url: {} for url in PATCHED_GIT}
    for manifest in _cargo_manifests():
        relative: str = str(manifest.relative_to(REPO_ROOT))
        for where, entry in _git_sources(tomllib.loads(manifest.read_text()).get("patch", {}).get("crates-io", {}), f"{relative}:patch.crates-io"):
            if entry["git"] in revs:
                revs[entry["git"]].setdefault(str(entry.get("rev")), set()).add(where)
    problems: list[str] = [f"{url}: {by_rev}" for url, by_rev in revs.items() if len(by_rev) > 1 or "None" in by_rev]
    assert not problems, "each patched egui repository needs one `rev` across all manifests:\n" + "\n".join(problems)
    assert revs[PATCHED_GIT[0]], "no [patch.crates-io] entry pins the egui family"


def _constant(script: Path, name: str) -> str:
    match: re.Match[str] | None = re.search(rf'^{name}: str = "([^"]+)"$', script.read_text(), flags=re.MULTILINE)
    assert match is not None, f"{script.relative_to(REPO_ROOT)}: {name} constant not found"
    return match.group(1)


def test_patched_rerun_checkouts_use_the_conda_pin_tag() -> None:
    scripts: tuple[Path, ...] = (CANONICAL_PREPARE, WEB_VIEWER_BUILD)
    tags: dict[str, str] = {str(script.relative_to(REPO_ROOT)): _constant(script, "RERUN_TAG") for script in scripts}
    commits: dict[str, str] = {str(script.relative_to(REPO_ROOT)): _constant(script, "RERUN_COMMIT") for script in scripts}
    assert set(tags.values()) == {_conda_rerun_version()}, f"RERUN_TAG must be the conda rerun-sdk pin: {tags}"
    assert len(set(commits.values())) == 1, f"the copies must check out the same commit: {commits}"


def test_rerun_patch_paths_name_the_conda_pin_tag() -> None:
    # cargo accepts a path patch of any version for a git dependency, so a stale checkout left on disk would still build.
    checkout: str = f"/rerun-{_conda_rerun_version()}/"
    paths: dict[str, str] = {}
    for manifest in _cargo_manifests():
        relative: str = str(manifest.relative_to(REPO_ROOT))
        patches: dict[str, Any] = tomllib.loads(manifest.read_text()).get("patch", {}).get(RERUN_GIT, {})
        paths |= {f"{relative}:{name}": entry["path"] for name, entry in patches.items() if "path" in entry}
    assert paths, f"no Cargo.toml patches {RERUN_GIT} to a prepared checkout"
    stale: dict[str, str] = {where: path for where, path in paths.items() if checkout not in path}
    assert not stale, f"[patch.{RERUN_GIT!r}] paths must point into a {checkout} checkout: {stale}"


def test_canonical_viewer_builds_one_patched_rerun() -> None:
    # A Rerun crate an extension names but the [patch] list misses comes from the unpatched tag: a second copy of Rerun.
    lock: Path = REPO_ROOT / "packages/canonical-viewer/rust/Cargo.lock"
    unpatched: list[str] = [package["name"] for package in tomllib.loads(lock.read_text())["package"] if package.get("source", "").startswith(f"git+{RERUN_GIT}")]
    assert not unpatched, f"add these to [patch.{RERUN_GIT!r}] in packages/canonical-viewer/rust/Cargo.toml: {unpatched}"
