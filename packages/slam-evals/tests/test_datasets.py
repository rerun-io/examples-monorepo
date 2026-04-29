"""Smoke checks for the per-dataset spec registry."""

from __future__ import annotations

from rerun.components import ViewCoordinates as ViewCoordinatesComponent

from slam_evals.data.datasets import DATASETS, EUROC, KITTI, DatasetSpec, lookup


def test_lookup_returns_known_specs() -> None:
    assert lookup("EUROC") is EUROC
    assert lookup("KITTI") is KITTI


def test_lookup_returns_none_for_unknown_dataset() -> None:
    assert lookup("NOT-A-REAL-DATASET") is None


def test_all_specs_appear_in_registry() -> None:
    """Every module-level spec should be reachable via ``lookup`` by name."""
    for spec in DATASETS:
        assert lookup(spec.name) is spec


def test_specs_are_frozen_dataclasses() -> None:
    """``DatasetSpec`` is frozen so registry instances stay hashable+immutable."""
    sample = DATASETS[0]
    try:
        sample.name = "MUTATED"  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or isinstance(exc, AttributeError)
    else:
        raise AssertionError("DatasetSpec should be frozen")


def test_world_view_coordinates_are_rerun_objects_or_none() -> None:
    for spec in DATASETS:
        assert spec.world_view_coordinates is None or isinstance(spec.world_view_coordinates, ViewCoordinatesComponent)


def test_dataset_spec_default_camera_convention() -> None:
    sample = DatasetSpec(name="X")
    assert sample.camera_convention == "RDF"
