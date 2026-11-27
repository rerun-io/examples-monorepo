"""The patched-checkout preparation: patches applied in order, the tree reused until the commit or a patch changes."""

import io
import tarfile
from pathlib import Path

from canonical_viewer.apis.prepare_rerun import PACKAGE_DIR, RERUN_PATCHES, RERUN_TAG, prepare_tree


def tar_of(files: dict[str, str]) -> bytes:
    buffer: io.BytesIO = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, text in files.items():
            data: bytes = text.encode()
            info: tarfile.TarInfo = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def write_patch(path: Path, old: str, new: str) -> Path:
    path.write_text(f"diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-{old}\n+{new}\n")
    return path


def test_patches_apply_in_order_and_the_tree_is_reused(tmp_path: Path) -> None:
    fetches: list[int] = []

    def archive() -> bytes:
        fetches.append(1)
        return tar_of({"tree/src/lib.rs": "one\n"})

    destination: Path = tmp_path / "patch" / "tree"
    patches: list[Path] = [write_patch(tmp_path / "1.patch", "one", "two"), write_patch(tmp_path / "2.patch", "two", "three")]
    assert prepare_tree(destination, archive, source_id="commit", patches=patches)
    assert (destination / "src/lib.rs").read_text() == "three\n"
    assert not prepare_tree(destination, archive, source_id="commit", patches=patches)
    assert len(fetches) == 1

    write_patch(patches[1], "two", "four")
    assert prepare_tree(destination, archive, source_id="commit", patches=patches)
    assert (destination / "src/lib.rs").read_text() == "four\n"
    assert prepare_tree(destination, archive, source_id="other-commit", patches=patches)
    assert len(fetches) == 3


def test_patches_name_the_rerun_tag_and_exist() -> None:
    for patch in RERUN_PATCHES:
        assert f"-{RERUN_TAG}-" in patch, f"{patch}: a Rerun patch names the tag it applies to"
        assert (PACKAGE_DIR / patch).is_file()
