"""Writer parity uses typed frozen fits; float32 base quaternions set a 1e-5° rotation floor.

Translation stays at 1e-6 m. The served writer composed float64 CSV poses before
serializing float32 matrices; the catalog base retains float32 quaternions.
Recomputation is checked separately at PR 4a residual/flag bands.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from frozen_v2 import GOLDEN_REFS, ROOT, SERVED, frozen_state, load_oak_detections, oak_median_reference
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation
from serde.json import from_json

import selfcap_curate.apis.calibrate as calibration_api
from selfcap_curate.apis.calibrate import Config, main
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.catalog_io import DATASET_NAME, open_segment
from selfcap_curate.device_tables import read_camera50_table
from selfcap_curate.factory import read_quest_block
from selfcap_curate.quest_fit import read_session_episode
from selfcap_curate.records import oak_quest_offset_ms, read_frozen_sync

pytestmark = pytest.mark.golden
SESSIONS = ("2025-11-24__027d8215", "2025-12-17__e0385cf5", "2025-12-30__65d00748", "2025-12-09__021d2e8b")


def payloads(path: Path) -> dict:
    result = {}
    for chunk in RrdReader(path).stream():
        entity = str(chunk.entity_path)
        if entity.startswith(("/world/gt", "/world/pred")) or "/pred/" in entity:
            continue
        for name, values in chunk.to_record_batch().to_pydict().items():
            if (
                name in ("rerun.controls.RowId", "log_time", "log_tick")
                or name.startswith(("quest_hands_", "kp3d_"))
                or name in ("version", "calib_phone")
            ):
                continue
            result.setdefault((entity, name), []).extend(values)
    return result


def compare_layers(actual: Path, expected: Path, unsynced_oak: str | None = None) -> tuple[float, float]:
    a, b = payloads(actual), payloads(expected)
    if unsynced_oak is not None:
        assert not any(entity == unsynced_oak for entity, _ in a)
        assert a["/__properties/calib", "oak_posed"] == [[False]]
        assert b["/__properties/calib", "oak_posed"] == [[True]]
        print("UNSYNCED_LAYER", actual.stem, len(b[unsynced_oak, "Transform3D:translation"]), 0, flush=True)
        b = {key: values for key, values in b.items() if key[0] != unsynced_oak}
        b["/__properties/calib", "oak_posed"] = [[False]]
    assert a.keys() == b.keys(), (a.keys() - b.keys(), b.keys() - a.keys())
    translation, degrees = 0.0, 0.0
    for (entity, component), values in a.items():
        reference = b[entity, component]
        if component == "Transform3D:translation":
            delta = float(
                np.max(
                    np.linalg.norm(
                        np.asarray(values, dtype=np.float64).reshape(-1, 3) - np.asarray(reference, dtype=np.float64).reshape(-1, 3), axis=1
                    )
                )
            )
            translation = max(translation, delta)
        elif component == "Transform3D:mat3x3":
            x = np.asarray(values, dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
            y = np.asarray(reference, dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
            delta = float(np.max(np.degrees(Rotation.from_matrix(y.transpose(0, 2, 1) @ x).magnitude())))
            degrees = max(degrees, delta)
        elif component.startswith("Pinhole:"):
            np.testing.assert_allclose(values, reference, rtol=1e-7, atol=1e-7, err_msg=f"{entity}:{component}")
        else:
            assert values == reference, (entity, component)
    return translation, degrees


@pytest.mark.parametrize("session", SESSIONS)
def test_calib_layer_matches_served_entity_values(session: str, tmp_path: Path, private_catalog, monkeypatch) -> None:
    sync_paths = sorted((ROOT / "v2/sync").glob(f"{session}__ep*.json"))
    record_path = ROOT / "calib/sessions" / f"{session}.json"
    if not sync_paths or not record_path.is_file():
        pytest.skip(f"calibration golden assets absent: {record_path}")
    syncs = [read_frozen_sync(path) for path in sync_paths]
    sources = {sync.key: sync for sync in syncs}
    borrowed = read_camera50_table().sessions.get(session)
    if borrowed:
        key = borrowed.healthy_segment.removeprefix("selfcap__")
        sources[key] = read_frozen_sync(ROOT / "v2/sync" / f"{key}.json")
    frozen = json.loads(record_path.read_text())
    maxima = [0.0, 0.0]
    with private_catalog(sources.values(), tmp_path, layers=False) as server:
        dataset = server.client().get_dataset(DATASET_NAME)
        segments = {sync.key: open_segment(dataset, sync.episode_key.segment) for sync in syncs}
        state = frozen_state(frozen, read_quest_block(segments[syncs[0].key]))
        for sync in syncs:
            output = tmp_path / "calib" / f"selfcap__{sync.key}.rrd"
            motion = read_session_episode(segments[sync.key], sync)
            write_calib_layer(output, state, motion)
            layout = motion.layout
            excluded = f"/world/rig_{layout.oak_rig:02d}" if oak_quest_offset_ms(sync) is None else None
            oak_median_reference(syncs, frozen["oak"]["median_px"])
            delta = compare_layers(output, SERVED / "calib-v2" / output.name, excluded)
            maxima = [max(x, y) for x, y in zip(maxima, delta, strict=True)]

        key_file = tmp_path / "keys.txt"
        key_file.write_text("\n".join(f"selfcap__{sync.key}" for sync in syncs))
        # OAK frozen reference remains test-only; phone observations come from sapiens2.

        monkeypatch.setattr(calibration_api, "pose_models", lambda: (None, None))
        monkeypatch.setattr(
            calibration_api,
            "detect_oak_hands_gpu",
            lambda _dataset, sync, *_args: load_oak_detections(GOLDEN_REFS / "oak-sapiens2" / sync.episode_key.rrd_name),
        )
        solved = main(
            Config(
                tmp_path / "solved",
                key_file=key_file,
                catalog_url=server.url(),
            )
        )
        restored = from_json(CalibSession, (tmp_path / "solved" / f"{session}.json").read_text())
        assert restored.session == session and restored.quest.mode == state.quest.mode
        for phone, expected in frozen["exo"]["phones"].items():
            result = solved.exo.phones[phone]
            assert result.median_px == pytest.approx(expected["median_px"], abs=0.5)
            for ep, values in expected["episodes"].items():
                assert result.episodes[ep].ok == values["ok"]
            pose = result.camera.world_T_cam
            reference = np.asarray(expected["world_T_cam"])
            angle = float(np.degrees(Rotation.from_matrix(reference[:3, :3].T @ pose[:3, :3]).magnitude()))
            centimetres = float(np.linalg.norm(reference[:3, 3] - pose[:3, 3]) * 100.0)
            print("CLI_FIT", session, phone, expected["median_px"], result.median_px, angle, centimetres, flush=True)
        assert solved.oak.ok == frozen["oak"]["ok"]
        assert solved.oak.median_px == pytest.approx(oak_median_reference(syncs, frozen["oak"]["median_px"]), abs=0.5)
        assert len(list((tmp_path / "solved").glob("*.rrd"))) == len(syncs)
    print("LAYER_MAX", session, *maxima, flush=True)
    assert maxima[0] <= 1e-6 and maxima[1] <= 1e-5, maxima
