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

Every dataset that appears under ``<benchmark_root>/`` gets an entry
here, even if we haven't classified its world frame yet. Datasets with
``world_view_coordinates=None`` skip the ``/world`` ViewCoordinates log
so the viewer falls back to its default — same effective behaviour as
having no entry at all, but explicit registration documents which
datasets we know about and which ones still need a convention pinned
down. Add the ``ViewCoordinates`` value the day you visually verify it.
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
# Visually verified against viewer playback. Names must match the on-disk
# dataset directory names exactly (``<benchmark_root>/<name>/<sequence>/...``).

EUROC = DatasetSpec(name="EUROC", world_view_coordinates=rr.ViewCoordinates.FLU)
KITTI = DatasetSpec(name="KITTI", world_view_coordinates=rr.ViewCoordinates.RDF)
TARTANAIR = DatasetSpec(name="TARTANAIR", world_view_coordinates=rr.ViewCoordinates.FRD)
RGBDTUM = DatasetSpec(name="RGBDTUM", world_view_coordinates=rr.ViewCoordinates.RDF)
SEVENSCENES = DatasetSpec(name="7SCENES", world_view_coordinates=rr.ViewCoordinates.RDF)
NUIM = DatasetSpec(name="NUIM", world_view_coordinates=rr.ViewCoordinates.RUB)
REPLICA = DatasetSpec(name="REPLICA", world_view_coordinates=rr.ViewCoordinates.RFU)
MSD = DatasetSpec(name="MSD", world_view_coordinates=rr.ViewCoordinates.LUF)


# ─── unclassified ───────────────────────────────────────────────────────────
# Every dataset that appears under ``<benchmark_root>/`` is declared here
# with ``world_view_coordinates=None`` until we visually verify and pin its
# convention. ``lookup`` still returns the spec, but the view_coordinates
# layer is skipped (see ``slam_evals.ingest.sequence.applicable_layers``)
# so the viewer falls back to its default world frame.
#
# When you classify one of these, move it up to the "confident specs"
# block and set the ``ViewCoordinates`` value.

ARIEL = DatasetSpec(name="ARIEL")
CAVES = DatasetSpec(name="CAVES")
DRUNKARDS = DatasetSpec(name="DRUNKARDS")
ETH = DatasetSpec(name="ETH")
HAMLYN = DatasetSpec(name="HAMLYN")
HILTI2022 = DatasetSpec(name="HILTI2022")
HILTI2026 = DatasetSpec(name="HILTI2026")
OPENLORIS = DatasetSpec(name="OPENLORIS")
OPENLORIS_D400 = DatasetSpec(name="OPENLORIS-D400")
OPENLORIS_T265 = DatasetSpec(name="OPENLORIS-T265")
ROVER = DatasetSpec(name="ROVER")
ROVER_D435I = DatasetSpec(name="ROVER-D435I")
ROVER_PICAM = DatasetSpec(name="ROVER-PICAM")
ROVER_T265 = DatasetSpec(name="ROVER-T265")
S3LI = DatasetSpec(name="S3LI")
SWEETCORALS = DatasetSpec(name="SWEETCORALS")
UT_CODA = DatasetSpec(name="UT-CODA")
VIDEOS = DatasetSpec(name="VIDEOS")
VITUM = DatasetSpec(name="VITUM")
YOUTUBE = DatasetSpec(name="YOUTUBE")


DATASETS: tuple[DatasetSpec, ...] = (
    # confident
    EUROC,
    KITTI,
    TARTANAIR,
    RGBDTUM,
    SEVENSCENES,
    NUIM,
    REPLICA,
    MSD,
    # unclassified (world_view_coordinates=None)
    ARIEL,
    CAVES,
    DRUNKARDS,
    ETH,
    HAMLYN,
    HILTI2022,
    HILTI2026,
    OPENLORIS,
    OPENLORIS_D400,
    OPENLORIS_T265,
    ROVER,
    ROVER_D435I,
    ROVER_PICAM,
    ROVER_T265,
    S3LI,
    SWEETCORALS,
    UT_CODA,
    VIDEOS,
    VITUM,
    YOUTUBE,
)

_BY_NAME: dict[str, DatasetSpec] = {d.name: d for d in DATASETS}


def lookup(name: str) -> DatasetSpec | None:
    """Return the ``DatasetSpec`` for ``name``, or ``None`` if unknown."""
    return _BY_NAME.get(name)
