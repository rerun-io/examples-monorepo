"""Fresh GPU calibration → local-file GPU keypoint CLI → one file."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from frozen_v2 import ENTITY, ROOT, SERVED, chunk_bytes, dense, oak_median_reference
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation

from selfcap_curate.apis.calibrate import Config as CalibrateConfig
from selfcap_curate.apis.calibrate import main as calibrate
from selfcap_curate.records import read_frozen_sync

pytestmark = pytest.mark.golden
SESSION = "2025-12-30__65d00748"
KEY = SESSION + "__ep001"


def test_session_keypoints_cli_preserves_fresh_calib_and_matches_served(tmp_path: Path, private_catalog):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for end-to-end calibration and RTMW golden")
    sync_paths = sorted((ROOT / "v2/sync").glob(f"{SESSION}__ep*.json"))
    reference_path = ROOT / "calib/sessions" / f"{SESSION}.json"
    if not sync_paths or not reference_path.is_file():
        pytest.skip(f"golden calibration assets missing: {reference_path}")
    syncs = [read_frozen_sync(p) for p in sync_paths]
    reference = json.loads(reference_path.read_text())
    with private_catalog(syncs, tmp_path / "base", layers=False) as server:
        assert ":51235" not in server.url()
        solved = calibrate(CalibrateConfig(tmp_path / "calib", segment_ids=tuple(f"selfcap__{s.key}" for s in syncs), catalog_url=server.url()))
        for phone, frozen in reference["exo"]["phones"].items():
            actual = solved.exo.phones[phone]
            assert actual.median_px == pytest.approx(frozen["median_px"], abs=0.5)
            assert {ep: fit.ok for ep, fit in actual.episodes.items()} == {ep: fit["ok"] for ep, fit in frozen["episodes"].items()}
            old, new = np.asarray(frozen["world_T_cam"]), actual.camera.world_T_cam
            angle = float(np.degrees(Rotation.from_matrix(old[:3, :3].T @ new[:3, :3]).magnitude()))
            cm = float(np.linalg.norm(old[:3, 3] - new[:3, 3]) * 100.0)
            print("E2E_POSE", phone, frozen["median_px"], actual.median_px, angle, cm, flush=True)
        assert solved.oak.ok == reference["oak"]["ok"]
        oak_reference = oak_median_reference(syncs, reference["oak"]["median_px"])
        assert solved.oak.median_px == pytest.approx(oak_reference, abs=0.5)
        print("E2E_OAK_MEDIAN", reference["oak"]["median_px"], oak_reference, solved.oak.median_px, flush=True)
        old_mount, new_mount = np.asarray(reference["oak"]["questrig_T_rgb"]), solved.oak.questrig_T_rgb
        assert new_mount is not None
        print(
            "E2E_OAK",
            float(np.degrees(Rotation.from_matrix(old_mount[:3, :3].T @ new_mount[:3, :3]).magnitude())),
            float(np.linalg.norm(old_mount[:3, 3] - new_mount[:3, 3]) * 100.0),
            flush=True,
        )
        calib = tmp_path / "calib" / f"selfcap__{KEY}.rrd"
        before = calib.read_bytes()
        key_file = tmp_path / "keys.txt"
        key_file.write_text(f"selfcap__{KEY}\n{KEY}\n")  # duplicate selection is idempotent
        command = [
            "pixi",
            "run",
            "-e",
            "selfcap-curate-dev",
            "--frozen",
            "selfcap-keypoints",
            "--calib-dir",
            str(calib.parent),
            "--output-dir",
            str(tmp_path / "complete"),
            "--catalog-url",
            server.url(),
            "--key-file",
            str(key_file),
            "--rate",
            "3.0",
        ]
        subprocess.run(command, cwd=Path(__file__).resolve().parents[3], check=True)
        final = tmp_path / "complete" / calib.name
        original, complete = chunk_bytes(calib), chunk_bytes(final)
        assert all(complete[key] == value for key, value in original.items())
        assert calib.read_bytes() == before
        assert len(list(final.parent.glob("*.rrd"))) == 1
        assert not any(str(c.entity_path).startswith("/world/gt") for c in RrdReader(final).stream())
        assert {str(c.entity_path) for c in RrdReader(final).stream() if str(c.entity_path).startswith("/__properties")} == {"/__properties/calib"}
        # Transform/pinhole preservation is exact, stronger than writer tolerances.
    served = SERVED / "calib-v2" / final.name
    stamps, xyz = dense(final, ENTITY, 3)
    expected_stamps, truth = dense(served, ENTITY, 3)
    np.testing.assert_array_equal(stamps, expected_stamps)
    distance = np.linalg.norm(xyz - truth, axis=-1)
    valid_change = float(np.mean(np.isfinite(xyz).all(2) != np.isfinite(truth).all(2)))
    print("E2E_3D", float(np.nanmedian(distance)), float(np.nanmax(distance)), valid_change, flush=True)
    assert np.nanmedian(distance) <= 0.010  # approved fresh-calibration solver-drift band
    overlays = [str(c.entity_path) for c in RrdReader(final).stream() if str(c.entity_path).endswith("/coco133_uv")]
    assert overlays
    for entity in sorted(set(overlays)):
        t, xy = dense(final, entity, 2)
        ts, uv = dense(served, entity, 2)
        np.testing.assert_array_equal(t, ts)
        delta = np.linalg.norm(xy - uv, axis=-1)
        assert np.isfinite(delta).any()
        print("E2E_2D", entity, float(np.nanmedian(delta)), float(np.nanmax(delta)), flush=True)
        assert np.nanmedian(delta) <= 1.0
