"""mamma Types: serde round-trips, the Rerun mapping structure, and to_rerun -> from_layer on catalog-shaped rows (unit lane)."""

from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from dagkit.testing import layer_table
from dagkit.time import STATIC
from dagkit.types import Log
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from mamma_fakes import assert_same
from serde import SerdeError, from_dict, to_dict
from simplecv.types.rig import RigCameras

from mamma.types import BodyTopology, Landmarks3d, MultiViewLandmarks, MultiViewTracks, SmplxFits

ROOT = "/world"
PEOPLE = "/world/mamma/people"
FLOATS = st.floats(-1e3, 1e3, width=32)


@st.composite
def tracks(draw: st.DrawFn) -> MultiViewTracks:
    v: int = draw(st.integers(1, 3))
    persons: list[int] = sorted(draw(st.sets(st.integers(0, 3), min_size=1, max_size=3)))
    rows: list[tuple[int, int]] = [(view, p) for view in range(v) for p in persons if draw(st.booleans())]  # may be empty (nobody tracked yet)
    k: int = len(rows)
    return MultiViewTracks(
        frame_index=draw(st.integers(0, 10**5)),
        views=tuple(f"rig_{i:02d}/cam_00" for i in range(v)),
        view=np.asarray([r[0] for r in rows], dtype=np.int64),
        person=np.asarray([r[1] for r in rows], dtype=np.int64),
        masks=torch.as_tensor(draw(arrays(np.bool_, (k, 6, 8)))),
        boxes_xyxy=draw(arrays(np.float32, (k, 4), elements=st.floats(1.0, 500.0, width=32))),
        scores=draw(arrays(np.float32, (k,), elements=st.floats(0.0, 1.0, width=32))),
        scale_xy=draw(arrays(np.float32, (v, 2), elements=st.floats(0.5, 4.0, width=32))),
    )


@st.composite
def landmarks(draw: st.DrawFn) -> MultiViewLandmarks:
    t: MultiViewTracks = draw(tracks())
    k: int = t.view.size
    n: int = 5 if k else 0  # an empty frameset has no landmark axis (LandmarksNode emits k = n = 0)
    tensor = lambda *shape: torch.as_tensor(draw(arrays(np.float32, shape, elements=FLOATS)))  # noqa: E731
    return MultiViewLandmarks(t.frame_index, t.views, t.view, t.person, tensor(k, n, 2), tensor(k, n), tensor(k, n), tensor(k, n), t.scale_xy)


@st.composite
def fits(draw: st.DrawFn) -> SmplxFits:
    persons: tuple[int, ...] = tuple(sorted(draw(st.sets(st.integers(0, 3), min_size=1, max_size=3))))
    p: int = len(persons)
    shape = lambda *s: draw(arrays(np.float32, (p, *s), elements=FLOATS))  # noqa: E731
    return SmplxFits(draw(st.integers(0, 10**5)), persons, shape(7, 3), shape(165), shape(16), shape(3))


@st.composite
def clouds(draw: st.DrawFn) -> Landmarks3d:
    persons: tuple[int, ...] = tuple(sorted(draw(st.sets(st.integers(0, 3), min_size=1, max_size=3))))
    valid: np.ndarray = draw(arrays(np.bool_, (len(persons), 6)))
    points: np.ndarray = np.where(valid[..., None], draw(arrays(np.float32, (len(persons), 6, 3), elements=FLOATS)), np.float32(0.0))
    return Landmarks3d(draw(st.integers(0, 10**5)), persons, points, valid)


@settings(max_examples=30, deadline=None)
@given(value=st.one_of(fits(), clouds()))
def test_host_types_serde_roundtrip_keeps_dtype_and_shape(value: SmplxFits | Landmarks3d) -> None:
    assert_same(from_dict(type(value), to_dict(value)), value)


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(SerdeError):
        from_dict(BodyTopology, {"persons": [0], "faces": [[0, 1, 2]], "colour": 1})


@settings(max_examples=30, deadline=None)
@given(value=tracks())
def test_tracks_map_to_engine_pixel_overlays_under_one_static_scale_per_view(value: MultiViewTracks) -> None:
    logs: list[Log] = value.to_rerun(ROOT)
    assert {log.path for log in logs if log.static} == {f"{ROOT}/{view}/pinhole/mamma" for view in value.views}
    masks: list[str] = [log.path for log in logs if "/masks/" in log.path]
    assert masks == [f"{ROOT}/{value.views[v]}/pinhole/mamma/masks/person_{p:02d}" for v, p in sorted(zip(value.view, value.person, strict=True))]
    assert all(not log.static for log in logs if log.path.endswith(("persons", *masks)))


@settings(max_examples=30, deadline=None)
@given(value=st.one_of(tracks(), landmarks(), fits(), clouds()), t_ns=st.integers(0, 10**12))
def test_to_rerun_then_from_layer_gives_the_value_back(value: object, t_ns: int) -> None:
    if isinstance(value, MultiViewTracks | MultiViewLandmarks):  # rows come back sorted by (view, person)
        order: np.ndarray = np.lexsort((value.person, value.view))
        value = type(value)(**{f.name: getattr(value, f.name)[order] if f.name not in ("frame_index", "views", "scale_xy") else getattr(value, f.name) for f in fields(value)})
    root: str = PEOPLE if isinstance(value, SmplxFits | Landmarks3d) else ROOT
    ((t_back, back),) = type(value).from_layer(layer_table([(t_ns, value)], root), root, "video_time")  # type: ignore[attr-defined]
    assert t_back == t_ns
    assert_same(back, value)


@settings(max_examples=30, deadline=None)
@given(first=tracks())
def test_tracks_from_layer_keeps_each_frames_masks(first: MultiViewTracks) -> None:
    second: MultiViewTracks = replace(first, frame_index=first.frame_index + 1, masks=~first.masks)
    values: list[tuple[int, MultiViewTracks]] = [(10, first), (20, second)]
    restored: list[tuple[int, MultiViewTracks]] = list(MultiViewTracks.from_layer(layer_table(values, ROOT), ROOT, "video_time"))
    for (t_ns, value), (t_back, back) in zip(values, restored, strict=True):
        assert t_back == t_ns
        assert_same(back, value)


def test_topology_is_static_per_person_and_reads_back() -> None:
    topology = BodyTopology((0, 2), np.arange(12, dtype=np.int32).reshape(4, 3))
    logs: list[Log] = topology.to_rerun(PEOPLE)
    assert [(log.path, log.static) for log in logs] == [(f"{PEOPLE}/person_{p:02d}{suffix}", True) for p in (0, 2) for suffix in ("", "/body/mesh")]
    sources = [batch.as_arrow_array().to_pylist() for log in logs if log.path.endswith(("_00", "_02")) for batch in log.data.as_component_batches()]
    assert sources == [["mamma_pred"], ["mamma_pred"]]  # the person node carries its source tag, as the GT people carry mosh_gt
    ((t_ns, back),) = BodyTopology.from_layer(layer_table([(STATIC, topology)], PEOPLE), PEOPLE, "video_time")
    assert t_ns == STATIC and back.persons == (0, 2)
    np.testing.assert_array_equal(back.faces, topology.faces)


def test_empty_framesets_read_back_as_empty_values() -> None:
    """Nobody tracked yet (before the bootstrap, or every track lost): the tick still round-trips, with its engine size."""
    none: np.ndarray = np.zeros(0, dtype=np.int64)
    views: tuple[str, ...] = ("rig_00/cam_00", "rig_01/cam_00")
    scale: np.ndarray = np.full((2, 2), 2.0, dtype=np.float32)
    empty = torch.zeros((0, 0))
    values: list[object] = [
        MultiViewTracks(7, views, none, none, torch.zeros((0, 48, 64), dtype=torch.bool), np.zeros((0, 4), np.float32), np.zeros(0, np.float32), scale),
        MultiViewLandmarks(7, views, none, none, empty[..., None].expand(0, 0, 2), empty, empty, empty, scale),
    ]
    for value in values:
        ((t_ns, back),) = type(value).from_layer(layer_table([(5, value)], ROOT), ROOT, "video_time")  # type: ignore[attr-defined]
        assert t_ns == 5
        assert_same(back, value)


@settings(max_examples=20, deadline=None)
@given(value=st.one_of(tracks(), landmarks()))
def test_views_come_from_the_mamma_overlays_not_from_a_rig_in_the_same_recording(value: MultiViewTracks | MultiViewLandmarks) -> None:
    """The file App logs the rig (``RigCameras.to_rerun``, which writes its own ``view_index``) beside the overlays."""
    order: np.ndarray = np.lexsort((value.person, value.view))
    value = type(value)(**{f.name: getattr(value, f.name)[order] if f.name not in ("frame_index", "views", "scale_xy") else getattr(value, f.name) for f in fields(value)})
    v: int = len(value.views)
    rig = RigCameras(value.views[::-1], np.tile(np.eye(3), (v, 1, 1)), np.tile(np.eye(4), (v, 1, 1)), np.tile(np.array([64, 48], dtype=np.int64), (v, 1)))
    ((_, back),) = type(value).from_layer(layer_table([(STATIC, rig), (3, value)], ROOT), ROOT, "video_time")
    assert_same(back, value)
