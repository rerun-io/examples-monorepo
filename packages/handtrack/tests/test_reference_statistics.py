"""Synthetic public contracts for reference frame streams and pooled statistics."""
import numpy as np
import pytest


def test_position_statistics_pool_samples_not_segment_quantiles():
    from handtrack.reference.results import position_statistics

    errors = np.concatenate((np.array([[10., np.nan]]), np.array([[30., 50.], [300., np.nan]])))
    posed = np.isfinite(errors)
    stats = position_statistics(errors, posed, np.ones((3, 2), dtype=bool))
    assert stats.median_mm == 40.
    assert stats.p90_mm == pytest.approx(225.)
    assert stats.below_20 == 1 / 6
    assert stats.below_50 == 2 / 6
    assert stats.wild == 1 / 4
    assert stats.scored == 4
    assert stats.posed_denominator == 4


def test_circles_restrict_errors_to_visible_pairs_and_count_wrong_views():
    from handtrack.reference.results import circle_statistics

    target = np.zeros((1, 2, 4, 3), dtype=np.float64)
    detected = target.copy()
    detected[0, :, :, 0] = [[3., 1000., 2000., 9.], [5., 1000., 2000., 11.]]
    detected[..., 2] = 2.
    visible = np.array([[[19, 18, 0, 21], [21, 1, 0, 19]]], dtype=np.int64)
    probability = np.array([[[.5, .9, .9, .8], [.6, .9, .5, .7]]])
    selected = np.ones((1, 2, 4), dtype=bool)
    stats = circle_statistics(detected, probability, target, visible, selected)
    assert stats[0].centre_median_px == 4.
    assert stats[0].centre_mean_px == 4.
    assert stats[0].centre_p90_px == pytest.approx(4.8)
    assert stats[0].radius_median_px == 2.
    assert stats[0].detection_rate == .5
    assert stats[0].visible_pairs == 2
    assert stats[1].error_pairs == 0
    assert stats[1].centre_mean_px is None
    assert stats[2].empty_pairs == 2
    assert stats[2].false_detection_rate == .5
    assert stats[2].selected == 2
    assert stats[2].selected_visible == 0
    assert stats[2].wrong_view == 2


def frame_stream(frames=2):
    from handtrack.reference.results import ReferenceFrames
    return ReferenceFrames(np.arange(frames, dtype=np.int64), np.zeros((frames, 1, 2)),
        np.ones((frames, 1, 2), dtype=bool), np.ones((frames, 2), dtype=bool),
        np.zeros((frames, 2, 4, 3)), np.zeros((frames, 2, 4)), np.zeros((frames, 2, 4, 3)),
        np.full((frames, 2, 4), 21, dtype=np.int64), np.zeros((frames, 1, 2, 4), dtype=bool))


def test_frame_stream_roundtrip_and_rejects_shapes_dtypes_keys(tmp_path):
    from dataclasses import fields

    from handtrack.reference.results import load_frames, save_frames
    stream = frame_stream()
    path = tmp_path / 'segment.npz'
    digest = save_frames(stream, path)
    loaded = load_frames(path, digest)
    np.testing.assert_array_equal(loaded.error_mm, stream.error_mm)
    np.testing.assert_array_equal(loaded.visible_landmarks, stream.visible_landmarks)
    for key, bad in [('posed', np.zeros((1, 1, 2), dtype=bool)),
                     ('error_mm', np.zeros((2, 2, 2))),
                     ('detnet_circle', np.zeros((2, 2, 3, 3))),
                     ('visible_landmarks', np.zeros((2, 2, 4), dtype=np.float64))]:
        arrays = {field.name: getattr(stream, field.name) for field in fields(stream)}
        arrays[key] = bad
        np.savez(path, **arrays)
        with pytest.raises(ValueError, match='Invalid'):
            load_frames(path)
    save_frames(stream, path)
    with pytest.raises(ValueError, match='digest'):
        load_frames(path, 'wrong')
    np.savez(path, error_mm=stream.error_mm)
    with pytest.raises(ValueError, match='keys'):
        load_frames(path)


def test_stream_requires_nan_for_unscored_errors(tmp_path):
    from dataclasses import fields

    from handtrack.reference.results import load_frames
    stream = frame_stream()
    arrays = {field.name: getattr(stream, field.name) for field in fields(stream)}
    arrays['posed'][0, 0, 0] = False
    arrays['error_mm'][0, 0, 0] = np.inf
    path = tmp_path / 'bad.npz'
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='errors'):
        load_frames(path)


def test_summary_pools_circle_samples_across_unequal_segments(tmp_path):
    from serde.json import from_json, to_json

    from handtrack.apis.reference_eval import Config, SegmentResult, publish_summary
    from handtrack.data.catalog import UMETRACK, SegmentInfo
    from handtrack.eval.segment import PositionScore
    from handtrack.reference.results import ReferenceMetrics, Summary, circle_statistics, save_frames

    infos = tuple(SegmentInfo(UMETRACK, f's{i}', 'real', 'hand_hand', 'testing', 'user_12', frames, 30)
                  for i, frames in enumerate((1, 3)))
    for index, info in enumerate(infos):
        stream = frame_stream((1, 3)[index])
        stream.detnet_circle[..., 0] = (10., 20.)[index]
        stream.detnet_circle[..., 2] = (2., 4.)[index]
        stream.detnet_presence[:] = .9
        stream.visible_landmarks[:, :, 1:] = 0
        stream.selected[:, 0, :, :2] = True
        cameras = circle_statistics(stream.detnet_circle, stream.detnet_presence, stream.gt_circle,
                                    stream.visible_landmarks, stream.selected[:, 0])
        metric = ReferenceMetrics(info.segment_id, 'detnet', 'none', 'id', len(stream.video_time_ns),
            PositionScore(0., None, None, 42 * len(stream.video_time_ns), 0), 2 * len(stream.video_time_ns),
            2 * len(stream.video_time_ns), 0, 1., 2 * len(stream.video_time_ns), (10., 20.)[index], (2., 4.)[index], cameras=cameras)
        npz = tmp_path / f'{info.segment_id}.npz'
        record = SegmentResult('id', info.segment_id, [metric], npz.name, save_frames(stream, npz))
        (tmp_path / f'{info.segment_id}.json').write_text(to_json(record))
    assert publish_summary(Config(output=tmp_path, modes=('detnet',)), infos, 'id')
    summary = from_json(Summary, (tmp_path / 'summary.json').read_text())
    score = summary.metrics[0]
    assert score.circle_samples == 8
    assert score.centre_error_px == 17.5
    assert score.radius_error_px == 3.5
    assert score.cameras[0].centre_median_px == 20.
    assert score.cameras[0].centre_p90_px == 20.
    assert score.cameras[0].radius_median_px == 4.
    assert score.cameras[0].detections == 8
    assert score.cameras[0].selected_visible == 8
    assert score.cameras[1].false_detections == 8
    assert score.cameras[1].empty_pairs == 8
    assert score.cameras[1].wrong_view == 8
    markdown = (tmp_path / 'summary.md').read_text()
    assert '| median mm | P90 mm | <20 | <50 | wild |' in markdown
    assert '| centre median px | centre mean px | centre P90 px |' in markdown
    assert '| 8/8 | 1.0000 |' in markdown
