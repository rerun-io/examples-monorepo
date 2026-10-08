"""The derived dataset keeps source images intact and reproduces the shared cube."""
import hashlib
import json
from pathlib import Path

import pytest

from gsplat_rust_renderer.apis.prepare_nerf_init import Config, initialization_ply, main


def test_prepare_links_source_and_adds_only_initialization(tmp_path: Path) -> None:
    source: Path = tmp_path / "source" / "lego"
    source.mkdir(parents=True)
    document: dict = {"camera_angle_x": 0.7, "frames": [{"file_path": "./train/r_0", "rotation": 0.0, "transform_matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}]}
    original: str = json.dumps(document)
    for split in ("train", "val", "test"):
        (source / split).mkdir()
        (source / f"transforms_{split}.json").write_text(original)
    output: Path = tmp_path / "derived"
    config: Config = Config(source_root=source.parent, output_root=output)
    main(config)
    main(config)
    assert (source / "transforms_train.json").read_text() == original
    assert json.loads((output / "lego/transforms_train.json").read_text()) == document | {"ply_file_path": "points3d.ply"}
    assert (output / "lego/train").resolve() == (source / "train").resolve()
    assert (output / "lego/points3d.ply").read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")


@pytest.mark.golden
def test_initialization_matches_shared_cube_bytes() -> None:
    assert hashlib.sha256(initialization_ply()).hexdigest() == "2ec0c12c4c6bf513914586490618e7359f0ce0bee5b5aa7adbf9a3c96a975ed3"
