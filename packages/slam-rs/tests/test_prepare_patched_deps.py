"""Preparation works from a Cargo cache without network access."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

from slam_rs.apis.prepare_patched_deps import CHANNEL_SHA256, PATCHED_CRATES, PatchedCrate, prepare


@pytest.fixture
def crate(tmp_path: Path) -> PatchedCrate:
    cache: Path = tmp_path / 'cargo/registry/cache/test'
    cache.mkdir(parents=True)
    archive: Path = cache / 'example-1.0.crate'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, content in [('Cargo.toml', '[package]\n'), ('hello.txt', 'before\n')]:
            entry: tarfile.TarInfo = tarfile.TarInfo(f'example-1.0/{name}')
            entry.size = len(content)
            tar.addfile(entry, io.BytesIO(content.encode()))
    patch: Path = tmp_path / 'change.patch'
    patch.write_text('diff --git a/hello.txt b/hello.txt\n--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n-before\n+after\n')
    crate: PatchedCrate = PatchedCrate('example', '1.0', hashlib.sha256(archive.read_bytes()).hexdigest(), 'change.patch')
    return crate


def test_applies_cached_archive(tmp_path: Path, crate: PatchedCrate) -> None:
    prepare(crate, tmp_path, tmp_path / 'cargo')
    assert (tmp_path / 'target/patch/example-1.0/hello.txt').read_text() == 'after\n'


def test_matching_inputs_are_a_noop(tmp_path: Path, crate: PatchedCrate) -> None:
    prepare(crate, tmp_path, tmp_path / 'cargo')
    output: Path = tmp_path / 'target/patch/example-1.0/hello.txt'
    before: int = output.stat().st_mtime_ns
    prepare(crate, tmp_path, tmp_path / 'cargo')
    assert output.stat().st_mtime_ns == before


def test_patch_change_reextracts_archive(tmp_path: Path, crate: PatchedCrate) -> None:
    prepare(crate, tmp_path, tmp_path / 'cargo')
    patch: Path = tmp_path / crate.patch
    patch.write_text(patch.read_text().replace('+after', '+changed'))
    prepare(crate, tmp_path, tmp_path / 'cargo')
    assert (tmp_path / 'target/patch/example-1.0/hello.txt').read_text() == 'changed\n'


def test_bad_archive_digest_is_refused(tmp_path: Path, crate: PatchedCrate) -> None:
    with pytest.raises(ValueError, match='archive SHA256 mismatch'):
        prepare(replace(crate, sha256='0' * 64), tmp_path, tmp_path / 'cargo')
    assert not (tmp_path / 'target/patch/example-1.0').exists()


def test_applies_inside_ignored_git_target(tmp_path: Path, crate: PatchedCrate) -> None:
    subprocess.run(['git', 'init', '-q', str(tmp_path.parent)], check=True)
    (tmp_path / '.gitignore').write_text('target/\n')
    prepare(crate, tmp_path, tmp_path / 'cargo')
    assert (tmp_path / 'target/patch/example-1.0/hello.txt').read_text() == 'after\n'


@pytest.mark.skipif(shutil.which('cargo') is None, reason='real patch smoke test requires cargo on PATH')
def test_real_patch_and_locked_cargo_resolution(tmp_path: Path) -> None:
    """The shipped patch matches the fork and Cargo uses the prepared crate."""
    package_dir: Path = Path(__file__).resolve().parents[1]
    cargo_home: Path = Path(os.environ.get('CARGO_HOME', str(Path.home() / '.cargo')))
    crate: PatchedCrate = PATCHED_CRATES[0]
    patch: Path = tmp_path / crate.patch
    patch.parent.mkdir(parents=True)
    shutil.copyfile(package_dir / crate.patch, patch)
    try:
        # A fresh destination exercises extraction and the real patch even when
        # the checkout already has a matching preparation marker.
        prepare(crate, tmp_path, cargo_home)
        prepare(crate, package_dir, cargo_home)
    except (urllib.error.URLError, TimeoutError) as error:
        pytest.skip(f'pinned crate archive is not cached and download is unavailable: {error}')
    stem: str = f'{crate.name}-{crate.version}'
    channel: Path = tmp_path / 'target/patch' / stem / 'src/device/handle/channel.rs'
    assert hashlib.sha256(channel.read_bytes()).hexdigest() == CHANNEL_SHA256
    result: subprocess.CompletedProcess[str] = subprocess.run(
        # CubeCL is optional; select its lane so it appears in the resolved graph.
        ['cargo', 'metadata', '--locked', '--offline', '--format-version', '1', '--features', 'slam-rs/gpu-wgpu'],
        cwd=package_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    # Raw JSON is intentional here: this is a contract test of Cargo's output.
    packages = [package for package in json.loads(result.stdout)['packages'] if package['name'] == crate.name]
    assert len(packages) == 1
    assert packages[0]['source'] is None
    assert Path(packages[0]['manifest_path']).resolve() == (package_dir / 'target/patch' / stem / 'Cargo.toml').resolve()
