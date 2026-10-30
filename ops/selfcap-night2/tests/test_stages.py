from pathlib import Path

import pytest
import rerun as rr
import stages


def recording(path, changed_rig=None):
    with rr.RecordingStream('geometry-test') as rec:
        rec.save(path)
        for rig in range(6):
            rec.log(f'/world/rig_{rig:02d}', rr.Transform3D(translation=[float(rig == changed_rig), 0.0, 0.0]), static=True)
        rec.flush()
    return path


@pytest.mark.parametrize('phones', [3, 4])
def test_geometry_guard_uses_session_wearables(tmp_path, phones):
    rigs = (f'/world/rig_{phones:02d}', f'/world/rig_{phones + 1:02d}')
    original = recording(tmp_path / 'original.rrd')
    oak_changed = recording(tmp_path / 'oak.rrd', phones)
    exo_changed = recording(tmp_path / 'exo.rrd', 0)
    signature = stages.ego_geometry_signature(original, rigs)
    assert stages.ego_geometry_signature(oak_changed, rigs) != signature
    assert stages.ego_geometry_signature(exo_changed, rigs) == signature


def test_geometry_guard_rejects_missing_wearable(tmp_path):
    path = recording(tmp_path / 'original.rrd')
    with pytest.raises(ValueError, match='missing'):
        stages.ego_geometry_signature(path, ('/world/rig_04', '/world/rig_06'))


@pytest.mark.parametrize('refined', [False, True])
def test_s2_restores_models_after_cache_load_failure(tmp_path, monkeypatch, refined):
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace
    import cache2d

    monkeypatch.setattr(stages, 'ROOT', tmp_path)
    work = tmp_path / 'runs/test/s1'
    (work / 'calib').mkdir(parents=True)
    recording(work / 'calib/test.rrd')
    (work / 'complete.json').write_text(json.dumps({'rate': 10.0}))
    @contextmanager
    def server(segment):
        yield SimpleNamespace(client=lambda: SimpleNamespace(get_dataset=lambda name: None))
    monkeypatch.setattr(stages, 'private', server)
    monkeypatch.setattr(stages, 'open_segment', lambda *args: SimpleNamespace(layout=SimpleNamespace(phones=('one', 'two'), oak_rig=3, quest_rig=4)))
    monkeypatch.setattr(stages, 'read_time_sync', lambda *args: SimpleNamespace(offsets_ms={'exo/one': 0, 'exo/two': 0}))
    decision_path = tmp_path / 'accepted-calibration.json'
    decision_path.write_text('{}')
    monkeypatch.setattr(stages, 'from_json', lambda *args: None)
    monkeypatch.setattr(stages, 'read_session_episode', lambda *args: None)
    monkeypatch.setattr(stages, 'write_calib_layer', lambda path, *args: path.write_bytes((work / 'calib/test.rrd').read_bytes()))
    original = (stages.keypoints.keypoint_models, stages.keypoints.detect_keypoints, stages.keypoint_geometry.select_wearers)
    with pytest.raises(ValueError, match='cache'):
        stages.s2('test', {'refined': refined, 'calib_path': str(decision_path)})
    assert (stages.keypoints.keypoint_models, stages.keypoints.detect_keypoints, stages.keypoint_geometry.select_wearers) == original
