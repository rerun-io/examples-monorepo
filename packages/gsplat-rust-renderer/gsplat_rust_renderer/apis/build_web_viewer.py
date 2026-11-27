"""Build the custom gsplat viewer for the web: the stock Rerun web viewer plus the compute splat renderer.

Steps: prepare a checkout of the Rerun release tag with the patched re_viewer, and the patched wgpu crate from its
verified crates.io archive (patches/*.patch, applied under target/patch/ and crates/gsplat-viewer-web/target/patch/), build crates/gsplat-viewer-web
for wasm32, run wasm-bindgen (output named re_viewer, like the stock viewer) and wasm-opt, then write
a viewer directory with the stock layout: the app files (index.html, re_viewer.js, re_viewer_bg.wasm)
and npm/ (index.js and the ES-module re_viewer.js that @rerun-io/web-viewer imports).
Runs in the gsplat-rust-renderer-web environment (wasm32 rust std, wasm-bindgen-cli, binaryen, clang).
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]
WEB_CRATE: Path = PACKAGE_DIR / "crates/gsplat-viewer-web"


@dataclass(frozen=True, slots=True)
class PatchedCrate:
    """A pinned crates.io archive and the patch applied to it."""

    name: str
    version: str
    sha256: str
    """Archive digest, as in Cargo.lock."""
    patch: str
    """Patch path relative to the package directory."""


PATCHED_CRATES: tuple[PatchedCrate, ...] = (
    PatchedCrate("wgpu", "30.0.0", "6d8f4bd44d92da5270f03409dba9f952dab24f128e05d6a554926101d1bf9114", "patches/wgpu-30.0.0-webgpu-subgroups.patch"),
)


# Rerun 0.44.0 is not on crates.io; the web crate's [patch."https://github.com/rerun-io/rerun"] points into a patched
# checkout of this tag.
# A staged copy of canonical_viewer.apis.prepare_rerun (tag, commit, fetch, prepare_tree): it goes when this web build
# moves onto the canonical viewer's web crate. ci/tests/test_rerun_pins.py keeps the tag and commit in step.
RERUN_URL: str = "https://github.com/rerun-io/rerun"
RERUN_TAG: str = "0.44.0"
RERUN_COMMIT: str = "f346ba2728b9f55afdfd5afdb2b976fecb98dcdd"
"""The tag's commit, checked after the fetch."""
RERUN_PATCH: str = "patches/re_viewer-0.44.0-web-hooks.patch"
"""Patch path relative to the package directory, applied to crates/top/re_viewer in the checkout."""


def prepare_tree(destination: Path, archive: Callable[[], bytes], *, source_id: str, patch: Path, patch_dir: str, tar_root: str) -> None:
    """Extract a tar archive, apply one patch and swap the tree into destination, unless an identical tree is there.

    Args:
        destination: Prepared tree; its name is the stem the archive and patch paths are relative to.
        archive: Fetches the tar archive; only called when the tree is missing or stale.
        source_id: Archive digest or commit that, with the patch digest, fingerprints the tree.
        patch: Patch file applied with ``git apply -p1``.
        patch_dir: Directory of the patched files relative to the tree root ("" for the root).
        tar_root: Directory relative to the tree's parent that the archive extracts into ("" when the archive's
            entries already start with the stem).
    """
    stem: str = destination.name
    fingerprint: str = f"{source_id}\n{hashlib.sha256(patch.read_bytes()).hexdigest()}\n"
    marker: Path = destination / ".prepared-sha256"
    if marker.is_file() and marker.read_text() == fingerprint:
        print(f"{stem}: unchanged")
        return
    data: bytes = archive()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{stem}-", dir=destination.parent) as temporary:
        staging: Path = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(staging / tar_root, filter="data")
        # Keep git from discovering the enclosing worktree (it would skip paths under the ignored target/).
        subprocess.run(
            ["git", "apply", "-p1", f"--directory={PurePosixPath(stem, patch_dir)}", str(patch)],
            cwd=staging,
            env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(staging.parent)),
            check=True,
        )
        (staging / stem / ".prepared-sha256").write_text(fingerprint)
        if destination.exists():
            destination.rename(staging / "previous")
        (staging / stem).rename(destination)
    print(f"{stem}: prepared")


def prepare(crate: PatchedCrate, cargo_home: Path) -> None:
    """Extract and patch one crate into the web crate's target/patch/, unless an identical tree is there."""
    stem: str = f"{crate.name}-{crate.version}"

    def fetch() -> bytes:
        cached: Path | None = next((cargo_home / "registry/cache").glob(f"*/{stem}.crate"), None)
        if cached is not None:
            archive: bytes = cached.read_bytes()
        else:
            with urllib.request.urlopen(f"https://static.crates.io/crates/{crate.name}/{stem}.crate", timeout=60) as response:
                archive = response.read()
        if hashlib.sha256(archive).hexdigest() != crate.sha256:
            raise ValueError(f"{stem}: archive SHA256 mismatch")
        return archive

    prepare_tree(WEB_CRATE / "target/patch" / stem, fetch, source_id=crate.sha256, patch=PACKAGE_DIR / crate.patch, patch_dir="", tar_root="")


def prepare_rerun(cargo_home: Path) -> None:
    """Check out the Rerun tag and patch re_viewer into the package's target/patch/, unless an identical tree is there."""
    stem: str = f"rerun-{RERUN_TAG}"

    def fetch() -> bytes:
        with tempfile.TemporaryDirectory(prefix=f".{stem}-fetch-") as temporary:
            # Rerun keeps test snapshots in LFS; git archive would download them through the smudge filter.
            git_env: dict[str, str] = dict(os.environ, GIT_CEILING_DIRECTORIES=temporary, GIT_LFS_SKIP_SMUDGE="1")
            # Cargo's own clone of the tag (the native build fetched it) saves the download.
            repository: Path | None = next(
                (
                    db
                    for db in sorted((cargo_home / "git/db").glob("rerun-*"))
                    if subprocess.run(["git", f"--git-dir={db}", "cat-file", "-e", f"{RERUN_COMMIT}^{{commit}}"], capture_output=True).returncode == 0
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
                    raise ValueError(f"{stem}: tag {RERUN_TAG} is {fetched}, expected {RERUN_COMMIT}")
            return subprocess.run(["git", f"--git-dir={repository}", "archive", "--format=tar", RERUN_COMMIT], env=git_env, check=True, capture_output=True).stdout

    prepare_tree(
        PACKAGE_DIR / "target/patch" / stem,
        fetch,
        source_id=RERUN_COMMIT,
        patch=PACKAGE_DIR / RERUN_PATCH,
        patch_dir="crates/top/re_viewer",
        tar_root=stem,
    )


def npm_module(no_modules_js: str) -> str:
    """Turn wasm-bindgen's no-modules output into the ES module with a default factory that npm/index.js imports."""
    head: str = "let wasm_bindgen = (function(exports) {"
    tail: str = "return Object.assign(__wbg_init, { initSync }, exports);\n})({ __proto__: null });"
    if not no_modules_js.startswith(head) or tail not in no_modules_js:
        raise ValueError("unexpected wasm-bindgen no-modules layout")
    body: str = no_modules_js[len(head) : no_modules_js.rindex(tail)]
    # deinit() drops the module and every cached memory view, as the stock npm build does.
    resets: list[str] = ["wasmModule", "wasm", *sorted(set(re.findall(r"^\s*let (cached\w*Memory\w*)\b", body, flags=re.MULTILINE)))]
    return (
        "\nexport default function() {\nconst exports = { __proto__: null };\n"
        + body
        + "\nfunction deinit() {\n  __wbg_init.__wbindgen_wasm_module = null;\n"
        + "".join(f"  {name} = null;\n" for name in resets)
        + "}\n\nreturn Object.assign(__wbg_init, { initSync, deinit }, exports);\n}\n"
    )


@dataclass
class BuildWebViewerConfig:
    """Build the web viewer and write it in the stock /rerun-viewer/<version>/ layout."""

    stock_dir: Path
    """A stock Rerun 0.44.0 web viewer directory (index.html, npm/index.js, ...) to copy the shell files from."""
    out_dir: Path
    """Output viewer directory, e.g. .../rerun-viewer/0.44.0-gsplat; replaced if it exists."""
    cargo_home: Path = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    """Cargo home with the crates.io archive cache and Cargo's git clones."""
    wasm_opt: bool = True
    """Run wasm-opt -O2 (strips debug info and names; about a minute)."""
    toolchain_prefix: Path | None = None
    """Prefix whose ``bin/clang`` and ``bin/ar`` compile the C deps for wasm32 (default ``$CONDA_PREFIX``; a rattler-build
    recipe passes its build prefix)."""


def main(config: BuildWebViewerConfig) -> None:
    """Prepare patches, build, bind, optimise and lay out the web viewer."""
    prepare_rerun(config.cargo_home)
    for crate in PATCHED_CRATES:
        prepare(crate, config.cargo_home)
    prefix: Path = config.toolchain_prefix or Path(os.environ["CONDA_PREFIX"])
    # Without CONDA_BUILD: re_build_tools (re_viewer's build script) treats a conda build as Rerun's own packaging and
    # runs `cargo metadata` on the patched crate, which fails outside Rerun's workspace.
    env: dict[str, str] = {name: value for name, value in os.environ.items() if name != "CONDA_BUILD"}
    env.update(
        CC_wasm32_unknown_unknown=f"{prefix}/bin/clang",
        AR_wasm32_unknown_unknown=f"{prefix}/bin/ar",
        CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="--cfg=web_sys_unstable_apis",
        CARGO_TARGET_DIR=str(WEB_CRATE / "target"),
    )
    start: float = time.perf_counter()
    subprocess.run(["nice", "-n", "10", "cargo", "build", "--release", "--locked", "--target", "wasm32-unknown-unknown"], cwd=WEB_CRATE, env=env, check=True)
    print(f"cargo build: {time.perf_counter() - start:.0f} s")
    wasm: Path = WEB_CRATE / "target/wasm32-unknown-unknown/release/gsplat_viewer_web.wasm"
    with tempfile.TemporaryDirectory(prefix="gsplat-web-bindgen-") as temporary:
        bindgen: Path = Path(temporary)
        subprocess.run(
            ["wasm-bindgen", str(wasm), "--out-dir", str(bindgen), "--out-name", "re_viewer", "--target", "no-modules", "--no-typescript"],
            check=True,
        )
        bg: Path = bindgen / "re_viewer_bg.wasm"
        if config.wasm_opt:
            start = time.perf_counter()
            optimised: Path = bindgen / "re_viewer_bg.opt.wasm"
            subprocess.run(["wasm-opt", "-O2", "--enable-bulk-memory", "--enable-nontrapping-float-to-int", "--enable-sign-ext",
                            "--enable-mutable-globals", "--enable-reference-types", "--enable-multivalue", "--strip-debug",
                            str(bg), "-o", str(optimised)], check=True)
            print(f"wasm-opt: {time.perf_counter() - start:.0f} s")
            bg = optimised
        if config.out_dir.exists():
            shutil.rmtree(config.out_dir)
        shutil.copytree(config.stock_dir, config.out_dir, ignore=shutil.ignore_patterns("SHA256SUMS", "re_viewer.js", "re_viewer_bg.wasm"))
        js: str = (bindgen / "re_viewer.js").read_text()
        (config.out_dir / "re_viewer.js").write_text(js)
        shutil.copyfile(bg, config.out_dir / "re_viewer_bg.wasm")
        (config.out_dir / "npm/re_viewer.js").write_text(npm_module(js))
        shutil.copyfile(bg, config.out_dir / "npm/re_viewer_bg.wasm")
    sums: list[str] = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(config.out_dir)}" for p in sorted(config.out_dir.rglob("*")) if p.is_file()]
    (config.out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    for path in config.out_dir.rglob("*"):
        path.chmod(0o775 if path.is_dir() else 0o644)
    print(f"wrote {config.out_dir} (re_viewer_bg.wasm {(config.out_dir / 're_viewer_bg.wasm').stat().st_size / 1e6:.1f} MB)")
