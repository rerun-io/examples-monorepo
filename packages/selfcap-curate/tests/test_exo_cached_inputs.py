"""A cached redecision must use the explicit original observation recording."""

import numpy as np
import pytest
import rerun as rr
from dataforge.hands import log_keypoints2d

from selfcap_curate.apis.exo_label import Config
from selfcap_curate.exo_refine_catalog import read_rtmw_recording


def test_cached_cli_requires_original_rtmw_recording(tmp_path):
    with pytest.raises(ValueError, match="rtmw"):
        Config("episode", tmp_path/'calib.json', tmp_path/'out', cached_labels=tmp_path/'labels.json')


def test_cached_cli_defaults_to_gate_v3_and_forwards_fold_rule(monkeypatch, tmp_path):
    import selfcap_curate.exo_label as module

    config = Config('episode', tmp_path/'calib.json', tmp_path/'out',
                    cached_labels=tmp_path/'labels.json', rtmw_rrd=tmp_path/'original.rrd')
    assert config.thresholds.sapiens_max_regression_px == 5.0
    assert config.thresholds.max_nonconverged_cv_folds == 1
    assert config.thresholds.require_full_fit_convergence
    calls = []

    def capture(args, output, segment, timeout_s, thresholds):
        calls.append(args)
        return module.LabelResult(segment, 'rejected', 'test', 0, 0.0, thresholds, gate_version=module.GATE_VERSION)

    monkeypatch.setattr(module, 'bounded_command', capture)
    result = module.label_session(config.segment, config.calib_record, config.output_dir,
                                  cached_labels=config.cached_labels, rtmw_rrd=config.rtmw_rrd)
    assert result.gate_version == 'gate-v3'
    assert calls[0][calls[0].index('--thresholds.sapiens-max-regression-px') + 1] == '5.0'
    assert calls[0][calls[0].index('--thresholds.max-nonconverged-cv-folds') + 1] == '1'


def test_original_rtmw_reader_preserves_weak_points_and_physical_rigs(tmp_path):
    path = tmp_path/'original.rrd'
    xy = np.ones((2, 133, 2), dtype=np.float32)
    scores = np.full((2, 133), .2, dtype=np.float32)
    with rr.RecordingStream('test', recording_id='episode', send_properties=False) as recording:
        recording.save(path)
        for rig in (0, 2):
            log_keypoints2d(recording, f'/world/rig_{rig:02d}/cam_00/pinhole/pred/rtmw/coco133_uv',
                            times_ns=np.array([1, 2], dtype=np.int64), frame_indices=None,
                            positions=xy + rig, confidence=scores)
        recording.flush()
    actual = read_rtmw_recording(path, 'episode', (2, 0))
    np.testing.assert_array_equal(actual[0].xy, xy + 2)
    np.testing.assert_array_equal(actual[1].xy, xy)
    np.testing.assert_array_equal(actual[0].scores, scores)
    with pytest.raises(ValueError, match="recording"):
        read_rtmw_recording(path, 'wrong-episode', (0,))
    with pytest.raises(ValueError, match="rig_01"):
        read_rtmw_recording(path, 'episode', (1,))
