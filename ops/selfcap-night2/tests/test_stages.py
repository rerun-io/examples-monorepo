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
