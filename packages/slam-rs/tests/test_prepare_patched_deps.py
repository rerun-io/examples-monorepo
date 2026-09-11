"""Preparation works from a Cargo cache without network access."""

import hashlib
import io
import subprocess
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from slam_rs.apis.prepare_patched_deps import PatchedCrate, prepare


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
