"""Named golden episodes only; private catalogs, never enumerate segments."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from frozen_v2 import FrozenFineLinks as GyroReference
from serde import serde
from serde.json import from_json

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_episode
from selfcap_curate.episode_links import measure_links
from selfcap_curate.records import Lag, read_sync

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parent / "fixtures"
KEYS = ("2025-11-24__027d8215__ep002", "2025-12-30__65d00748__ep001", "2025-12-08__cc808233__ep001")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class VideoStarts:
    oak_left: float
    """Source OAK left MP4 start in seconds, measured once with ffprobe."""
    oak_rgb: float
    """Source OAK RGB MP4 start in seconds."""
    quest_video: float
    """Source Quest right MP4 start in seconds."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ScratchFinalLink:
    served_ms: float
    """Frozen lag served by the corpus, before any catalog clock correction."""
    scratch_final_ms: float
    """Final scratch rerun on the same measurement window and widths."""
    drift_ms: float
    """Recorded absolute drift between the served and final scratch lags."""


@serde
@dataclass(frozen=True, slots=True)
class WindowReference:
    window_s: tuple[float, float]
    """Frozen scratch measurement interval."""
    video_start_s: VideoStarts
    """Raw MP4 starts; applied only to video-to-CSV expected lags."""
    video_start_source: str
    """ffprobe command and source paths used to freeze the starts."""
    search_half_width_s: dict[str, float]
    """Frozen per-link coarse search widths."""
    search_half_width_source: str
    """Provenance of the frozen widths."""
    scratch_final: dict[str, ScratchFinalLink] = field(default_factory=dict)
    """Final scratch references where frozen served values no longer reproduce."""
    scratch_final_source: str | None = None
    """Diagnostic provenance for the recorded scratch revision drift."""


@pytest.mark.parametrize("key", KEYS)
def test_catalog_links_match_frozen_measurements(key: str, tmp_path: Path, private_catalog) -> None:
    reference = read_sync(FIXTURES / f"{key}.sync.json")
    with private_catalog([reference], tmp_path, layers=False) as server:
        inputs = read_episode(open_segment(server.client().get_dataset(DATASET_NAME), f"selfcap__{key}"))
    window = from_json(WindowReference, (FIXTURES / f"{key}.links.json").read_text())
    print("REFERENCE", window.video_start_source, window.search_half_width_source, window.scratch_final_source, flush=True)
    result = measure_links(inputs, window_s=window.window_s, search_half_width_s=window.search_half_width_s)
    gyro_offset = (
        None
        if result.left_to_gyro.lag_s is None or result.quest_video_to_gyro.lag_s is None
        else (result.quest_video_to_gyro.lag_s - result.left_to_gyro.lag_s) * 1000.0
    )
    pose_offset = (
        None
        if result.lag_left_to_pose.lag_s is None or result.lag_quest_video_to_pose.lag_s is None
        else (result.lag_quest_video_to_pose.lag_s - result.lag_left_to_pose.lag_s) * 1000.0
    )
    print(f"{key}: Quest gyro={gyro_offset} ms; old pose-chain={pose_offset} ms", flush=True)
    mismatches: list[str] = []
    video_corrections: dict[str, float] = {
        name: source_start_s - float(video.t_ns[0]) * 1e-9
        for name, source_start_s, video in zip(
            ("oak_left", "oak_rgb", "quest_video"),
            (window.video_start_s.oak_left, window.video_start_s.oak_rgb, window.video_start_s.quest_video),
            inputs.videos,
            strict=True,
        )
    }
    print("VIDEO_CORRECTIONS", key, video_corrections, flush=True)
    corrections: dict[str, float] = {
        "lag_left_to_pose": video_corrections["oak_left"],
        "lag_left_to_gyro": video_corrections["oak_left"],
        "lag_quest_video_to_pose": video_corrections["quest_video"],
        "lag_rgb_to_left": video_corrections["oak_rgb"] - video_corrections["oak_left"],
        "lag_left_to_quest_video": video_corrections["oak_left"] - video_corrections["quest_video"],
    }
    for name in ("lag_left_to_pose", "lag_rgb_to_left", "lag_quest_video_to_pose", "lag_left_to_gyro", "lag_gyro_to_pose"):
        measured: Lag = getattr(result, name)
        frozen: Lag = getattr(reference.links, name)
        if frozen.lag_s is None:
            continue
        baseline_s: float = frozen.lag_s
        if name in window.scratch_final:
            baseline: ScratchFinalLink = window.scratch_final[name]
            assert frozen.lag_s * 1000.0 == pytest.approx(baseline.served_ms, abs=1e-6), f"{name}: served fixture changed"
            assert abs(baseline.served_ms - baseline.scratch_final_ms) == pytest.approx(baseline.drift_ms, abs=0.1), (
                f"{name}: scratch revision drift changed"
            )
            baseline_s = baseline.scratch_final_ms / 1000.0
            print("DRIFT", key, name, baseline.served_ms, baseline.scratch_final_ms, baseline.drift_ms, flush=True)
        correction_s: float = corrections.get(name, 0.0)
        expected_s: float = baseline_s + correction_s
        delta_s: float | None = None if measured.lag_s is None else abs(measured.lag_s - expected_s)
        print("COMPARISON", key, name, frozen.lag_s, baseline_s, correction_s, expected_s, measured.lag_s, delta_s, flush=True)
        if measured.lag_s != pytest.approx(expected_s, abs=0.002):
            mismatches.append(f"{name}: {measured.lag_s} vs {expected_s}, residual={delta_s} s")
    fine = from_json(GyroReference, (FIXTURES / f"{key}.links_rgb.json").read_text())
    if not fine.edge:
        assert result.left_to_gyro is not None
        assert result.rgb_to_gyro is not None
        for name, measured, frozen_s, correction_s in (
            ("fine_left_to_gyro", result.left_to_gyro, fine.left_to_gyro_s, video_corrections["oak_left"]),
            ("fine_rgb_to_gyro", result.rgb_to_gyro, fine.rgb_to_gyro_s, video_corrections["oak_rgb"]),
        ):
            expected_s = frozen_s + correction_s
            delta_s = None if measured.lag_s is None else abs(measured.lag_s - expected_s)
            print("COMPARISON", key, name, frozen_s, frozen_s, correction_s, expected_s, measured.lag_s, delta_s, flush=True)
            if measured.lag_s != pytest.approx(expected_s, abs=0.002):
                mismatches.append(f"{name}: {measured.lag_s} vs {expected_s}, residual={delta_s} s")
    print(key, "frozen Quest offset_ms", reference.offsets_ms["quest_video"])
    assert not mismatches, "; ".join(mismatches)
    assert gyro_offset is not None
