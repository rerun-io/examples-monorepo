from dataclasses import replace

import numpy as np

from selfcap_curate.exo_label import GateThresholds, accepts, filter_labels, grid_crop
from selfcap_curate.exo_landmarks import Landmark
from selfcap_curate.exo_refine_catalog import Gate, Score


def test_gate_requires_independent_metrics_and_limits_hands():
    before = {name: Score(50, 20.0, 30.0) for name in ('headset', 'phones', 'sapiens_hands', 'rtmw_hands', 'body_transfer')}
    after = {name: Score(50, 10.0, 20.0) for name in before}
    gate = Gate(before, after, (), False)
    assert accepts(gate)
    assert not accepts(replace(gate, candidate={**after, 'rtmw_hands': Score(50, 21.51, 30.0)}))
    assert not accepts(replace(gate, candidate={**after, 'body_transfer': Score(0, None, None)}))
    assert not accepts(replace(gate, candidate={**after, 'phones': Score(50, 19.5, 30.0)}))


def test_crop_uses_source_coordinates_and_filters_unknown_targets():
    image = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    crop, bounds = grid_crop(image, 1900.0, 50.0, 240, 'view=0 quest0')
    assert bounds == (1680, 0, 1920, 240)
    assert crop.shape == (540, 480, 3)
    labels = [Landmark(0, 1.0, 'quest0', 1800.0, 60.0, .9), Landmark(0, 1.0, 'quest1', 1800.0, 60.0, .9), Landmark(0, 1.0, 'quest0', 1800.0, 60.0, .2)]
    assert filter_labels(labels, {(0, 1.0, 'quest0'): bounds}) == labels[:1]


def test_timestamp_rounding_keeps_correct_label_and_normalizes_time():
    label = Landmark(3, 3.731750224, 'phone1', 829.0, 319.0, .8)
    allowed = {(3, 3.7317502240000002, 'phone1'): (595, 70, 1135, 610)}
    result = filter_labels([label], allowed)
    assert len(result) == 1
    assert result[0].video_time == 3.7317502240000002
    assert filter_labels([replace(label, video_time=3.8)], allowed) == []


def test_static_phone_anchor_is_shared_but_headset_stays_held_out():
    from selfcap_curate.exo_calibration import Camera
    from selfcap_curate.exo_landmarks import Landmarks
    from selfcap_curate.exo_refine import RefinementProblem, select_labels

    labels = [Landmark(0, 0.0, 'phone1', 10.0, 20.0, .8), Landmark(0, 0.0, 'quest0', 40.0, 20.0, .8),
              Landmark(0, 1.0, 'quest0', 50.0, 20.0, .8)]
    problem = RefinementProblem((Camera(np.zeros(11)),), (), Landmarks(labels, 'test', 'exact', '2026-10-30'), np.zeros((3, 3)), np.tile(np.eye(3), (3, 1, 1)))
    fold = select_labels(problem, 1, static_phones=True)
    assert [(label.video_time, label.target_id) for label in fold.landmarks.labels] == [(0.0, 'phone1'), (1.0, 'quest0')]
    assert [label.target_id for label in select_labels(problem, 1).landmarks.labels] == ['quest0']


def test_deadline_returns_fallback_record(tmp_path):
    from selfcap_curate.exo_label import bounded_command

    result = bounded_command(['--help'], tmp_path, 'synthetic', .001)
    assert result.decision == 'timeout'
    assert (tmp_path / 'result.json').exists()
    assert not (tmp_path / 'calib.json').exists()


def test_hand_thresholds_are_explicit_and_independent():
    before = {name: Score(50, 20.0, 30.0) for name in ('headset', 'phones', 'sapiens_hands', 'rtmw_hands', 'body_transfer')}
    after = {name: Score(50, 10.0, 20.0) for name in before}
    after['rtmw_hands'] = Score(50, 21.3, 30.0)
    gate = Gate(before, after, (), False)
    assert accepts(gate)
    assert not accepts(gate, GateThresholds(rtmw_max_regression_px=1.0))
    assert not accepts(replace(gate, candidate={**after, 'sapiens_hands': Score(50, 21.1, 30.0)}))


def test_device_vp_table_is_typed_and_has_all_measured_devices():
    from selfcap_curate.device_vp import read_vp_table

    table = read_vp_table()
    assert len(table.devices) == 8
    assert table.devices['Dragon-413A7CC1'].focal_px == 888.7573979270643
    assert table.devices['Titanium-57BB9E1A'].focal_px == 894.9294280422055
    assert all(device.low_px <= device.focal_px <= device.high_px for device in table.devices.values())
