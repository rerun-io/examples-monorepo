"""MAMMA SMPL-X forward against the release: why a template fit may use v1.1, and the held-out Vicon markers (eval Extra)."""

import os
from pathlib import Path

import numpy as np
import pytest
from conftest import raw_asset
from scipy.spatial import KDTree

from dataforge import paths
from dataforge.datasets.mamma_body import BODY_FILES, MODEL_FILES, PosedPerson, SmplxBody
from dataforge.datasets.mamma_source import PersonFit, SequenceLabels, read_eval_labels

pytestmark = pytest.mark.golden

MAMMA_ROOT: Path = Path(os.environ.get("DATAFORGE_MAMMA_ROOT", "/mnt/nas/datasets/mamma"))
MODEL_ROOT: Path = Path(os.environ.get("DATAFORGE_MAMMA_MODEL_ROOT", str(paths.raw_root() / "mamma-body-models")))
EXTRA: str = "mamma_eval_extra/070525_BedlamLab_Eval_Solo_Dancing_00202_1"
DANCE: str = "mamma_eval_dance/250225_WestCoastSwing_Basic_Whip_03675_03676_1"


def body() -> SmplxBody:
    for model in BODY_FILES:
        raw_asset("MAMMA SMPL-X model (dataforge-download mamma)", model.local(MODEL_ROOT))
    return SmplxBody(MODEL_ROOT)


def test_a_template_fit_does_not_depend_on_the_model_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    # A subject template replaces the model's template and zero betas leave its shape basis unused: the only parts in which
    # the locked head differs from v1.1. So the gendered (Singles) template fits are exact with v1.1 male and female.
    labels: SequenceLabels = read_eval_labels(raw_asset("MAMMA eval Dance GT", MAMMA_ROOT / DANCE / "gt/global.npz"))
    person: PersonFit = labels.people[0]
    assert labels.method == "mosh_v_template" and person.model == "locked_head_neutral" and not person.betas.any()
    frames: np.ndarray = np.arange(0, labels.frames, 15, dtype=np.int64)
    locked: PosedPerson = body().pose(person, frames)
    monkeypatch.setitem(MODEL_FILES, "locked_head_neutral", MODEL_FILES["v1_1_neutral"])
    v1_1: PosedPerson = SmplxBody(MODEL_ROOT).pose(person, frames)
    assert np.abs(locked.vertices - v1_1.vertices).max() < 1e-5
    assert np.nanmax(np.abs(locked.coco133 - v1_1.coco133)) < 1e-5


def test_the_held_out_vicon_markers_sit_on_the_extra_mesh() -> None:
    # MoSh++ fit the body to 73 markers; these 37 were held out. A marker sits on the skin or clothing, plus its own radius.
    labels: SequenceLabels = read_eval_labels(raw_asset("MAMMA eval Extra GT", MAMMA_ROOT / EXTRA / "gt/global.npz"))
    assert labels.method == "mosh_betas"
    frames: np.ndarray = np.arange(0, labels.frames, 30, dtype=np.int64)  # multiples of the mesh stride: every frame keeps its mesh
    posed: PosedPerson = body().pose(labels.people[0], frames)
    markers: np.ndarray = np.load(raw_asset("MAMMA eval Extra markers", MAMMA_ROOT / EXTRA / "markers/vicon_m37.npy"))[frames]
    distances_mm: np.ndarray = np.stack([KDTree(vertices).query(frame)[0] for vertices, frame in zip(posed.vertices, markers, strict=True)]) * 1000
    print(f"\nheld-out marker to nearest vertex: median {np.median(distances_mm):.1f} mm, p95 {np.percentile(distances_mm, 95):.1f} mm")
    assert np.median(distances_mm) < 25.0
