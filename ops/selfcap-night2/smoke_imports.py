"""Fail deployment before launching workers if stage interfaces do not match."""
from pathlib import Path
import inspect
import stages
from selfcap_curate.apis import align, calibrate, keypoints, exo_label
from selfcap_curate import keypoints as geometry

root = Path(__file__).resolve().parents[2]
for module in (align, calibrate, keypoints, exo_label, geometry):
    assert root in Path(module.__file__).resolve().parents, module.__file__
    if module is not geometry:
        assert callable(module.main)
        assert inspect.isclass(module.Config)
assert callable(stages.s1) and callable(stages.s2)
assert callable(geometry.select_wearers)
assert callable(keypoints.detect_keypoints) and callable(keypoints.keypoint_models)
assert 'rtmw_rrd' in exo_label.Config.__dataclass_fields__
assert 'cached_labels' in exo_label.Config.__dataclass_fields__
print('S1, S2 and label stage imports match pinned package sources:', root)

from selfcap_curate.exo_label import DEFAULT_THRESHOLDS, GATE_VERSION
assert GATE_VERSION == "gate-v3"
assert DEFAULT_THRESHOLDS.sapiens_max_regression_px == 5.0
assert DEFAULT_THRESHOLDS.rtmw_max_regression_px == 1.5
assert DEFAULT_THRESHOLDS.max_nonconverged_cv_folds == 1
assert DEFAULT_THRESHOLDS.require_full_fit_convergence
print("Gate-v3 thresholds and mandatory full-fit convergence verified")
