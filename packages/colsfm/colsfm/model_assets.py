"""Where the uncommitted model graphs live, and what to run when they are not there.

`packages/colsfm/data/models/` is gitignored: the RaCo graphs are exported, not shipped, and
the engines built from them are specific to one TensorRT version and one GPU
architecture. A fresh clone therefore has none of it — and since 2026-09-06 the
default run asks for RaCo-ALIKED and LightGlue+ (NOTES.md decision 17), so "none of
it" is the first thing a fresh clone hits.

The pipeline **stops** there rather than falling back to `pycolmap`. A silent
fallback would be the one failure mode worth avoiding: every number a run reports —
keypoints, matches, reprojection, ATE — moves with the extractor, so a run that
quietly changed backend would be a different experiment wearing the same name. The
message names both ways out: build the graphs, or ask for the ablation.

The path constants live here rather than beside the code that runs them because
naming a file has no business importing TensorRT: `colsfm.features_raco` and
`colsfm.matching_trt` both pull in `colsfm.tensorrt_runtime`, which imports
`tensorrt` and `cuda.bindings` at module scope, and `colsfm.run_config` has to be
able to check for a missing graph without any of that.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from colsfm import DATA_DIR
from colsfm.features import FeatureBackend
from colsfm.matching import FLEXGLUE_FP8_COMMAND, FLEXGLUE_MODEL_DIR, FLEXGLUE_MODEL_FILES, FlexGlueOptions, MatchingBackend, gpu_has_fp8

RACO_ONNX_PATH: Final[Path] = DATA_DIR / "models" / "raco-aliked-b1-16.onnx"
"""The batch-dynamic RaCo-ALIKED graph; its engine is cached beside it.

Not committed — `packages/colsfm/data/models` is gitignored and the graph is 8.9 MB of
weights. Produce it with
`pixi run -e colsfm colsfm-raco-export`."""

RACO_DYNAMIC_ONNX_PATH: Final[Path] = DATA_DIR / "models" / "raco-aliked-dyn.onnx"
"""The batch- **and** shape-dynamic RaCo-ALIKED graph; its engine is cached beside it.

Also not committed. Produce it with
`pixi run -e colsfm colsfm-raco-export`.

Not in `required_raco_graphs`: it is only reached by an image whose size group the
fixed-shape engine does not cover (`colsfm.features_raco.select_raco_engine`), which
Galileo, at the profile maximum, never is. A dataset that does need it says so at the
engine build, by name."""

RACO_LIGHTGLUE_ONNX_PATH: Final[Path] = DATA_DIR / "models" / "raco" / "aliked_lightglue" / "lightglue_aliked.onnx"
"""The RaCo-trained LightGlue+ graph; its engine is cached beside it.

Not committed either. Produce it with `pixi run -e colsfm colsfm-raco-export`, which writes
both this and the batch-1 extractor under `packages/colsfm/data/models/raco/aliked_lightglue/`."""

RACO_EXPORT_COMMANDS: Final[tuple[str, ...]] = (
    "pixi run -e colsfm colsfm-raco-export",
    "pixi run -e colsfm colsfm-download-flexglue",
    FLEXGLUE_FP8_COMMAND,
)
"""The commands that produce every graph `required_raco_graphs` asks for: the RaCo export, the FlexGlue download and
the FlexGlue FP8 calibration."""

PYCOLMAP_ABLATION_FLAGS: Final[str] = "--features-backend pycolmap --matching-backend pycolmap"
"""The documented no-GPU-engine path: COLMAP's own ALIKED and LightGlue, downloaded by COLMAP."""


def required_raco_graphs(features_backend: FeatureBackend, matching_backend: MatchingBackend) -> tuple[Path, ...]:
    """The RaCo graphs one pair of backend choices cannot run without.

    Args:
        features_backend: `--features-backend`.
        matching_backend: `--matching-backend`.

    Returns:
        The graphs, in stage order; empty unless a `raco` backend was asked for.
    """
    graphs: list[Path] = []
    if features_backend == "raco":
        graphs.append(RACO_ONNX_PATH)
    if matching_backend == "raco":
        graphs.append(RACO_LIGHTGLUE_ONNX_PATH)
    if matching_backend in ("flexglue", "flexglue_ort"):
        graphs.extend(FLEXGLUE_MODEL_DIR / name for name in FLEXGLUE_MODEL_FILES.values())
    if matching_backend == "flexglue" and gpu_has_fp8():  # below sm_89 the matcher runs its prepared graph, no FP8 file
        graphs.append(FLEXGLUE_MODEL_DIR / FlexGlueOptions().fp8_file())  # a run keeps FlexGlue's default thresholds
    return tuple(graphs)


def missing_raco_graphs_message(missing: Sequence[Path]) -> str:
    """What a run says when the default backend's graphs are not in this checkout.

    Args:
        missing: The graphs `required_raco_graphs` named and the disk does not have.

    Returns:
        One message naming the files, the export that produces them, and the
        ablation that needs neither.
    """
    files: str = "\n".join(f"  {path}" for path in missing)
    exports: str = "\n".join(f"  {command}" for command in RACO_EXPORT_COMMANDS)
    return (
        "[colsfm] the default feature and matching backends are RaCo-ALIKED and LightGlue+, "
        f"and this checkout is missing:\n{files}\n"
        f"Export them (they are gitignored, so every clone builds its own):\n{exports}\n"
        f"Or run the no-GPU-engine ablation, which needs no graph of ours: {PYCOLMAP_ABLATION_FLAGS}"
    )


def check_raco_graphs(features_backend: FeatureBackend, matching_backend: MatchingBackend) -> None:
    """Stop a run whose backends need a graph this checkout does not have.

    Called from `colsfm.run_config.resolve_run`, i.e. before the workspace exists,
    so a fresh clone pays nothing to find out.

    Args:
        features_backend: `--features-backend`.
        matching_backend: `--matching-backend`.

    Raises:
        FileNotFoundError: When a required graph is missing, with the export command
            and the ablation named.
    """
    missing: tuple[Path, ...] = tuple(path for path in required_raco_graphs(features_backend, matching_backend) if not path.is_file())
    if missing:
        raise FileNotFoundError(missing_raco_graphs_message(missing))
