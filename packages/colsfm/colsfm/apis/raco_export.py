"""Export every RaCo graph the default run needs into `packages/colsfm/data/models/` (the `colsfm-raco-export` task)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from colsfm.model_assets import RACO_DYNAMIC_ONNX_PATH, RACO_LIGHTGLUE_ONNX_PATH, RACO_ONNX_PATH
from colsfm.raco_export import ExportConfig, export_batched_extractor, export_extractor, export_matcher


@dataclass
class Config:
    """Which graphs to (re)write."""

    export: ExportConfig = field(default_factory=ExportConfig)
    """Shape contract of the graphs; the defaults are what the run's engines are profiled for."""
    force: bool = False
    """Rewrite graphs that already exist."""


def main(config: Config) -> None:
    """Write the LightGlue+ matcher, the batch-1 and batch-dynamic extractors, and the shape-dynamic extractor.

    Args:
        config: Export settings.
    """
    torch.set_grad_enabled(False)
    matcher_dir: Path = RACO_LIGHTGLUE_ONNX_PATH.parent
    matcher_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Path, Callable[[Path], None]]] = [
        ("LightGlue+ matcher", RACO_LIGHTGLUE_ONNX_PATH, lambda path: export_matcher(config.export, path)),
        ("batch-1 extractor", matcher_dir / "aliked.onnx", lambda path: export_extractor(config.export, path)),
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
        export(path)
