"""The web build's patched-tree step: extract, patch, mark, and skip an unchanged tree."""

import io
import tarfile
from pathlib import Path

from gsplat_rust_renderer.apis.build_web_viewer import prepare_tree

PATCH: str = """--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1 +1 @@
-pub fn version() -> u32 { 1 }
+pub fn version() -> u32 { 2 }
"""


def _archive(root: str) -> bytes:
    """A tar with src/lib.rs under root ("" for no root directory)."""
    buffer: io.BytesIO = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        content: bytes = b"pub fn version() -> u32 { 1 }\n"
        info: tarfile.TarInfo = tarfile.TarInfo(str(Path(root, "crate/src/lib.rs")))
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_prepare_tree_patches_once_and_reuses_an_identical_tree(tmp_path: Path) -> None:
    patch: Path = tmp_path / "fix.patch"
    patch.write_text(PATCH)
    destination: Path = tmp_path / "target/patch/thing-1.0.0"
    fetches: list[str] = []

    def fetch() -> bytes:
        fetches.append("fetch")
        return _archive("")

    for _ in range(2):
        prepare_tree(destination, fetch, source_id="abc", patch=patch, patch_dir="crate", tar_root=destination.name)
    assert fetches == ["fetch"]
    assert (destination / "crate/src/lib.rs").read_text() == "pub fn version() -> u32 { 2 }\n"
    assert (destination / ".prepared-sha256").read_text().startswith("abc\n")
    assert sorted(p.name for p in destination.parent.iterdir()) == ["thing-1.0.0"]

    prepare_tree(destination, lambda: _archive("thing-1.0.0"), source_id="def", patch=patch, patch_dir="crate", tar_root="")
    assert (destination / ".prepared-sha256").read_text().startswith("def\n")
    assert (destination / "crate/src/lib.rs").read_text() == "pub fn version() -> u32 { 2 }\n"
    assert sorted(p.name for p in destination.parent.iterdir()) == ["thing-1.0.0"]
