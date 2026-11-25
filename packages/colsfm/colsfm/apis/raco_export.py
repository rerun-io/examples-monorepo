"""Export every RaCo graph the default run needs into `colsfm.MODEL_DIR` (the `colsfm-raco-export` task)."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from colsfm.model_assets import RACO_DYNAMIC_ONNX_PATH, RACO_LIGHTGLUE_ONNX_PATH, RACO_ONNX_PATH
from colsfm.raco_export import ExportConfig, export_batched_extractor, export_matcher


@dataclass
class Config:
    """Which graphs to (re)write."""

    export: ExportConfig = field(default_factory=ExportConfig)
    """Shape contract of the graphs; the defaults are what the run's engines are profiled for."""
    force: bool = False
    """Rewrite graphs that already exist."""


def main(config: Config) -> None:
    """Write the LightGlue+ matcher, the batch-dynamic extractor and the shape-dynamic extractor.

    Each graph is exported to a temporary file beside its destination and renamed into place, so an interrupted or
    failed export never leaves a partial graph that a later run would keep as "exists".

    Args:
        config: Export settings.
    """
    torch.set_grad_enabled(False)
    matcher_dir: Path = RACO_LIGHTGLUE_ONNX_PATH.parent
    matcher_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Path, Callable[[Path], None]]] = [
        ("LightGlue+ matcher", RACO_LIGHTGLUE_ONNX_PATH, lambda path: export_matcher(config.export, path)),
        ("batch-dynamic extractor", RACO_ONNX_PATH, lambda path: export_batched_extractor(config.export, path)),
        (
            "batch- and shape-dynamic extractor",
            RACO_DYNAMIC_ONNX_PATH,
            lambda path: export_batched_extractor(dataclasses.replace(config.export, dynamic_shape=True), path),
        ),
    ]
    for label, path, export in jobs:
        if path.is_file() and not config.force:
            print(f"[colsfm] {label}: {path} exists, kept (--force rewrites)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[colsfm] exporting the {label} to {path}")
        staged: Path = path.with_name(f".{path.stem}.{os.getpid()}.onnx")
        try:
            export(staged)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)
