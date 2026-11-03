"""Session CLI rejects unsafe selections/outputs before catalog or GPU work."""

from pathlib import Path

import pytest

from selfcap_curate.apis.keypoints import Config, main


def test_keypoints_cli_rejects_invalid_session_and_existing_targets(tmp_path: Path):
    calib = tmp_path / "calib"
    calib.mkdir()
    key = "2025-12-30__65d00748__ep001"
    with pytest.raises(ValueError, match="explicit session episode keys"):
        main(Config(calib, tmp_path / "out", rate=3.0))
    with pytest.raises(ValueError, match="exactly one session"):
        main(Config(calib, tmp_path / "out", rate=3.0, segment_ids=(key, "2025-12-20__25aeb66a__ep001")))
    with pytest.raises(FileNotFoundError):
        main(Config(calib, tmp_path / "out", rate=3.0, segment_ids=(key,)))
    (calib / f"selfcap__{key}.rrd").touch()
    output = tmp_path / "out"
    output.mkdir()
    target = output / f"selfcap__{key}.rrd"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        main(Config(calib, output, rate=3.0, segment_ids=(key,)))
    assert target.read_bytes() == b"keep"
    with pytest.raises(ValueError):
        main(Config(calib, calib / "nested", rate=3.0, segment_ids=(key,)))
    with pytest.raises(ValueError):
        main(Config(calib, Path("/mnt/nas/datasets/forbidden-keypoints"), rate=3.0, segment_ids=(key,)))






def test_rate_is_optional_but_an_explicit_rate_is_still_checked(tmp_path):
    calib = tmp_path / "calib"
    calib.mkdir()
    key = "2025-12-30__65d00748__ep001"
    (calib / f"selfcap__{key}.rrd").touch()
    assert Config(calib, tmp_path / "out", segment_ids=(key,)).rate is None
    with pytest.raises(ValueError, match="rate must be finite and positive"):
        main(Config(calib, tmp_path / "out", rate=0.0, segment_ids=(key,)))


def test_keypoints_reads_the_named_dataset_like_calibrate(tmp_path, monkeypatch):
    from selfcap_curate.apis import keypoints as api

    seen = {}

    def stop(*_args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop before catalog access")

    monkeypatch.setattr(api, "open_session", stop)
    with pytest.raises(RuntimeError, match="stop before catalog access"):
        main(Config(tmp_path / "calib", tmp_path / "out", rate=3.0, dataset="selfcap-hard26-r2"))
    assert seen["dataset"] == "selfcap-hard26-r2"
