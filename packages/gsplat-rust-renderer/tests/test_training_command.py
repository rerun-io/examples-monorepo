"""Dataset routing and argument forwarding without shell interpolation."""
from pathlib import Path

import pytest

from gsplat_rust_renderer.apis.train import Config, main


@pytest.mark.parametrize("flags", [["--seed", "99"], ["--", "--seed", "99"]])
def test_training_uses_existing_colmap_data_and_preserves_extra_flags(tmp_path: Path, flags: list[str]) -> None:
    dataset = tmp_path / "data/tandt/truck"
    dataset.mkdir(parents=True)
    capture = tmp_path / "arguments"
    binary = tmp_path / "trainer"
    binary.write_text(
        "#!/usr/bin/env python3\nimport pathlib, sys\n"
        f"pathlib.Path({str(capture)!r}).write_text('\\n'.join(sys.argv[1:]))\n"
    )
    binary.chmod(0o755)
    main(Config(scene="truck", iterations=100, mode="video", binary=binary,
                data_root=tmp_path / "data", output_root=tmp_path / "runs"), flags)
    assert capture.read_text().splitlines() == [
        str(dataset), "--total-train-iters", "100", "--export-every", "7000",
        "--save", str(tmp_path / "runs/truck/100/video/training.rrd"),
        "--export-path", str(tmp_path / "runs/truck/100/video"), "--video", "--seed", "99",
    ]
