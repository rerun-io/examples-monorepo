"""Align explicit episodes of one session; catalog inputs, local outputs only."""

from dataclasses import dataclass
from pathlib import Path

from dataforge.timing import SequenceTimer

from selfcap_curate.apis import open_session, session_output
from selfcap_curate.catalog_io import CATALOG_URL, open_segment, read_catalog_hands, read_episode, read_session_clock
from selfcap_curate.detection_io import write_detections_layer
from selfcap_curate.episode_links import measure_links
from selfcap_curate.hands_gpu import detect_episode, pose_models
from selfcap_curate.patch_base import patch
from selfcap_curate.quest_fit import load_quest_hands
from selfcap_curate.records import SelfcapLayout, write_sync
from selfcap_curate.session_fit import EpisodeLinks, FitEpisode
from selfcap_curate.sync_policy import fit_session


@dataclass
class Config:
    """One session's explicit catalog ids, source bases, and output directory."""

    base_dir: Path
    """As-cut base RRD directory with presented timestamps, read only."""
    output_dir: Path
    """New session directory, published only when all outputs are complete."""
    segment_ids: tuple[str, ...] = ()
    """Explicit selfcap__date__sid8__epNNN ids."""
    key_file: Path | None = None
    """Optional text file, one segment id or unprefixed key per line."""
    catalog_url: str = CATALOG_URL
    """Catalog containing these fresh bases; never registered or swapped here."""


def main(config: Config) -> None:
    """Measure, detect, fit and patch one selected session without raw reads."""
    _, keys, dataset = open_session(
        config.segment_ids, config.key_file, config.catalog_url, output=config.output_dir, protected=(config.base_dir,), source=config.base_dir
    )
    timer = SequenceTimer()
    with session_output(config.output_dir) as output:
        with timer.stage("session/models"):
            detector, pose = pose_models()
        episodes: list[FitEpisode] = []
        layouts: list[SelfcapLayout] = []
        for key in keys:
            segment: str = key.segment
            with timer.stage(f"{key}/catalog"):
                handle = open_segment(dataset, segment)
                layouts.append(handle.layout)
                inputs = read_episode(handle)
                clock = read_session_clock(handle)
                duration: float = float(inputs.videos[0].t_ns[-1] - inputs.videos[0].t_ns[0]) * 1e-9
                window: tuple[float, float] = max(0.5, duration / 2.0 - 25.0), min(duration - 0.5, duration / 2.0 + 25.0)
            with timer.stage(f"{key}/links"):
                result = measure_links(inputs, window_s=window)
            with timer.stage(f"{key}/hands"):
                quest = load_quest_hands(read_catalog_hands(handle))
                detections = detect_episode(handle, quest, detector, pose)
                episodes.append(FitEpisode(clock.session, EpisodeLinks(key.key, result, window, clock.start_s), tuple(detections), detections, quest))
        with timer.stage("session/fit"):
            records = fit_session(episodes)
        for key, episode, layout in zip(keys, episodes, layouts, strict=True):
            record = records[key.key]
            with timer.stage(f"{key}/write_patch"):
                patch(config.base_dir / key.rrd_name, None, record, output_path=output / key.rrd_name, mode="fresh")
                write_sync(output / key.sync_name, record)
                write_detections_layer(output / "sapiens2" / key.rrd_name, record, layout, episode.detections)
    print("ALIGN_TIME", timer.stage_s, flush=True)
