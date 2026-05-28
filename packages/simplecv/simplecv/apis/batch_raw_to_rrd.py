from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import cast

import numpy as np
import rerun as rr
from tqdm import tqdm

from simplecv.apis.view_exoego import VisualizeConfig, visualize_exo_ego
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.rerun_log_utils import RerunTyroConfig

np.set_printoptions(suppress=True)


@dataclass
class BatchConvertConfig:
    """Configuration for batch converting annotated sequences to RRD files."""

    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset factory capable of producing the annotated ``BaseExoEgoSequence``."""
    rrd_save_dir: Path = Path("data/exoego-forge-catalog")
    """Output directory that will receive the generated ``.rrd`` files."""
    max_conversions: int | None = 5
    """Optional cap on how many sequences to convert; ``None`` processes all available episodes."""
    dry_run: bool = False
    """When ``True``, only report the sequences that would be converted without writing files."""
    force: bool = False
    """When ``True``, overwrite existing ``.rrd`` files instead of skipping them."""
    log_exo: bool = True
    """Enable exo-camera imagery, intrinsics, and projections during conversion."""
    log_ego: bool = True
    """Enable ego-camera imagery, intrinsics, and projections during conversion."""
    log_labels: bool = True
    """Enable COCO-133 label logging during conversion."""
    log_mano: bool = True
    """Enable derived MANO mesh/keypoint logging during conversion."""
    log_mano_vertex_normals: bool = False
    """Compute and log dynamic MANO mesh vertex normals during conversion."""


def main(config: BatchConvertConfig):
    start_time: float = timer()
    sequence_cls = cast(type[BaseExoEgoSequence], config.dataset._target)
    num_sequences: int = sequence_cls.num_sequences_for_config(config.dataset)
    if config.max_conversions is not None:
        num_sequences = min(num_sequences, config.max_conversions)

    seen_recording_ids: set[str] = set()
    seen_output_paths: set[Path] = set()
    sequence_iter = cast(Iterable[BaseExoEgoSequence], sequence_cls.iter_episode_sequences(config.dataset))

    for idx, current_exoego_sequence in enumerate(tqdm(sequence_iter, total=num_sequences, desc="Processing sequences")):
        identity: SequenceIdentity = current_exoego_sequence.sequence_identity
        rrd_save_path: Path = identity.rrd_path(config.rrd_save_dir)
        resolved_rrd_save_path: Path = rrd_save_path.expanduser().resolve()
        sequence_label: str = identity.sequence_key

        if identity.recording_id in seen_recording_ids:
            raise ValueError(f"Duplicate recording_id while converting sequences: {identity.recording_id}")
        if resolved_rrd_save_path in seen_output_paths:
            raise ValueError(f"Duplicate RRD output path while converting sequences: {rrd_save_path}")
        seen_recording_ids.add(identity.recording_id)
        seen_output_paths.add(resolved_rrd_save_path)

        if rrd_save_path.exists() and not config.force:
            tqdm.write(f"[skip-existing] {sequence_label} -> {rrd_save_path}")
        elif config.dry_run:
            tqdm.write(f"[dry-run] {sequence_label} -> {rrd_save_path}")
        else:
            rrd_save_path.parent.mkdir(parents=True, exist_ok=True)
            current_cfg = VisualizeConfig(
                rr_config=RerunTyroConfig(
                    application_id="exoego-forge",
                    recording_id=identity.recording_id,
                    save=rrd_save_path,
                ),
                dataset=current_exoego_sequence.config,
                log_exo=config.log_exo,
                log_ego=config.log_ego,
                log_labels=config.log_labels,
                log_mano=config.log_mano,
                log_mano_vertex_normals=config.log_mano_vertex_normals,
            )
            rec: rr.RecordingStream = current_cfg.rr_config.rec_stream
            rr.send_recording_name(identity.sequence_key, recording=rec)
            rec.send_property(
                "info",
                rr.AnyValues(
                    sequence_key=identity.sequence_key,
                    num_frames=len(current_exoego_sequence),
                    has_ego=current_exoego_sequence.ego_sequence is not None,
                    has_exo=current_exoego_sequence.exo_sequence is not None,
                ),
            )
            visualize_exo_ego(current_exoego_sequence, current_cfg)
            rec.flush(timeout_sec=600.0)

        if config.max_conversions is not None and idx + 1 >= config.max_conversions:
            break

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
