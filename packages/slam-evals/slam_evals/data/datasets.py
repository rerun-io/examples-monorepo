"""Per-dataset metadata catalog.

The VSLAM-LAB corpus ships ~30 datasets, each with its own world-frame
convention (gravity-aligned Z-up vs OpenGL Y-up vs NED, …). The convention
isn't derivable from the on-disk data — it's published in each dataset's
docs — so we encode the closed knowledge here.

Pattern: a ``DatasetSpec`` frozen dataclass + module-level instances per
dataset + a single tuple registry. Each dataset is one declaration; new
metadata fields (e.g. depth-factor defaults, eval masks, citations) are
added to the dataclass and per-dataset constructor calls without
scattering match-arms across N functions.

Unknown dataset names from disk fall through to ``None`` in ``lookup``;
the caller (``layer_calibration``) treats that as "skip the /world
ViewCoordinates log" so the viewer uses its default. That preserves
today's behavior for every dataset we haven't explicitly classified.
"""

from __future__ import annotations

from dataclasses import dataclass

import rerun as rr
from rerun.components import ViewCoordinates as ViewCoordinatesComponent


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Static knowledge about a VSLAM-LAB dataset's frame conventions.

    ``world_view_coordinates`` is the convention the dataset's GT trajectory
    is expressed in. Logged at ``/world`` so the 3D view orients correctly.
    Note the type: ``rr.ViewCoordinates.RDF`` and friends are *components*
    (``rerun.components.ViewCoordinates``), not the archetype class
    ``rr.ViewCoordinates`` which they're class-attribute-attached to.

    ``camera_convention`` is the per-image-plane axis convention. Almost
    universal "RDF" (X-right, Y-down, Z-forward); override only for OpenGL-
    style synthetic datasets. Currently informational — wired into
    ``Pinhole`` via ``simplecv.log_pinhole`` later when a non-RDF dataset
    needs it.
    """

    name: str
    world_view_coordinates: ViewCoordinatesComponent | None = None
    camera_convention: str = "RDF"


# ─── confident specs ────────────────────────────────────────────────────────
# Add a new entry here when a dataset's viewer rendering looks wrong;
# verify in the catalog after a `--layers calibration --force` re-ingest.
# Names must match the on-disk dataset directory names exactly (see manifest).

EUROC = DatasetSpec(name="EUROC", world_view_coordinates=rr.ViewCoordinates.FLU)
KITTI = DatasetSpec(name="KITTI", world_view_coordinates=rr.ViewCoordinates.RDF)
TARTANAIR = DatasetSpec(name="TARTANAIR", world_view_coordinates=rr.ViewCoordinates.FRD)
RGBDTUM = DatasetSpec(name="RGBDTUM", world_view_coordinates=rr.ViewCoordinates.RDF)
NUIM = DatasetSpec(name="NUIM", world_view_coordinates=rr.ViewCoordinates.RUB)
REPLICA = DatasetSpec(name="REPLICA", world_view_coordinates=rr.ViewCoordinates.RUB)


DATASETS: tuple[DatasetSpec, ...] = (
    EUROC,
    KITTI,
    TARTANAIR,
    RGBDTUM,
    NUIM,
    REPLICA,
)

_BY_NAME: dict[str, DatasetSpec] = {d.name: d for d in DATASETS}


def lookup(name: str) -> DatasetSpec | None:
    """Return the ``DatasetSpec`` for ``name``, or ``None`` if unknown."""
    return _BY_NAME.get(name)
