from pathlib import Path

import pytest

from selfcap_soma.batch import selected_segments


def test_segment_list_is_ordered_deduplicated_and_rejects_path_injection(tmp_path: Path) -> None:
    path = tmp_path / "list.txt"
    path.write_text("selfcap__2025-12-03__d31bf1da__ep001\n# comment\n")
    assert selected_segments(["selfcap__2025-12-03__d31bf1da__ep001"], path) == ["selfcap__2025-12-03__d31bf1da__ep001"]
    with pytest.raises(ValueError, match="segment"):
        selected_segments(["../../other"], None)


def test_protected_dataset_fails_before_any_catalog_or_transport(tmp_path: Path) -> None:
    from selfcap_soma.batch import Config, run_segment

    config = Config(dataset='dataforge-selfcap', spark_dir=Path('/tmp/soma'), output=tmp_path)
    with pytest.raises(ValueError, match='protected dataset'):
        run_segment(config, 'selfcap__2025-12-03__d31bf1da__ep001')
    assert not list(tmp_path.iterdir())


def test_missing_body_exits_with_documented_input_limitation(monkeypatch, capsys):
    from selfcap_soma import batch
    from selfcap_soma.shape_frames import InsufficientBodyObservation

    def unavailable(config, segment):
        raise InsufficientBodyObservation('no usable torso frames')

    monkeypatch.setattr(batch, 'run_segment', unavailable)
    config = batch.Config(dataset='test', spark_dir=Path('/remote'), segment=['selfcap__2025-12-04__5cb1e14c__ep001'])
    with pytest.raises(SystemExit) as raised:
        batch.main(config)
    assert raised.value.code == 20
    assert 'SOMA_INPUT_LIMITATION insufficient_body_observation' in capsys.readouterr().err
