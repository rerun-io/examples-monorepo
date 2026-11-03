"""``selfcap-preview``: write one preview layer file per segment; it neither ships nor registers them."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from beartype.roar import BeartypeException
from dataforge import paths, writing
from rerun.catalog import CatalogClient, DatasetEntry

from selfcap_curate.preview import PreviewOptions, PreviewResult, RigChoice, build_segment_preview
from selfcap_curate.records import episode_keys


@dataclass
class Config:
    """Write ``<output_dir>/<segment>.rrd`` for each selected segment."""

    catalog_url: str
    """Catalog that holds the segments; only read."""
    dataset: str
    """Dataset on ``catalog_url``."""
    output_dir: Path
    """Directory for the new layer files; an existing file is never overwritten."""
    segment_ids: tuple[str, ...] = ()
    """Explicit segment ids; with no ``key_file`` either, every segment of the dataset."""
    key_file: Path | None = None
    """Optional text file, one segment id or unprefixed key per line."""
    seconds: float = 30.0
    """Length of the middle window."""
    fps: float | None = None
    """Thin videos and body to about this rate; None keeps the source rate."""
    ego: RigChoice = None
    """OAK rig index; None reads it from the rig ``kind`` metadata."""
    exo: RigChoice = None
    """Exo phone rig index; None takes the first posed phone, rig 2 first."""
    workers: int = 3
    """Segments built at once; each encode also uses several CPU threads."""


def main(config: Config) -> None:
    """Write each segment's preview file; a failed segment prints one line, and the run exits nonzero at the end.

    Raises:
        SystemExit: If any segment failed.
    """
    if config.workers < 1:
        raise ValueError("workers must be at least 1")
    if config.seconds <= 0.0 or (config.fps is not None and config.fps <= 0.0):
        raise ValueError("seconds and fps must be positive")
    paths.require_outside([config.output_dir], roots=[*paths.NAS_RAW_ROOTS])
    entry: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    requested: list[str] = [key.segment for key in episode_keys(config.segment_ids, config.key_file)] if config.segment_ids or config.key_file else []
    segments: list[str] = writing.select_segments(entry, requested)
    options: PreviewOptions = PreviewOptions(seconds=config.seconds, fps=config.fps, ego=config.ego, exo=config.exo)

    def build(segment: str) -> tuple[str, bool]:
        started: float = perf_counter()
        try:
            result: PreviewResult = build_segment_preview(entry, segment, config.output_dir / f"{segment}.rrd", options)
        except BeartypeException:
            raise
        except Exception as error:
            print(f"preview FAIL {segment}: {' '.join(str(error).split())} ({int(perf_counter() - started)} s)", flush=True)
            return segment, False
        print(
            f"preview ok   {segment}: window {result.lo_ns / 1e9:.1f}-{result.hi_ns / 1e9:.1f} s, ego rig {result.ego_rig}, exo rig {result.exo_rig}, "
            f"camera model {'yes' if result.exo_camera_model else 'NO'}, moving {list(result.moving_rigs)}, exo {result.exo_frames} frames, ego {result.ego_frames} frames, "
            f"file {result.file_bytes / 1e6:.1f} MB ({int(perf_counter() - started)} s)",
            flush=True,
        )
        return segment, True

    with ThreadPoolExecutor(config.workers) as pool:
        results: list[tuple[str, bool]] = list(pool.map(build, segments))
    failed: list[str] = [segment for segment, built in results if not built]
    print(f"wrote {len(segments) - len(failed)}/{len(segments)} preview files to {config.output_dir}")
    if failed:
        raise SystemExit(f"preview failed for {len(failed)} of {len(segments)} segments")
