from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Final, cast


def _load_mhr70_module() -> ModuleType:
    try:
        from sam3d_body.metadata import mhr70

        return mhr70
    except ModuleNotFoundError as import_error:
        source_path: Path = Path(__file__).resolve().parents[4] / "sam3d-body-rerun" / "src" / "sam3d_body" / "metadata" / "mhr70.py"
        if not source_path.exists():
            raise ModuleNotFoundError(
                "Could not import sam3d_body.metadata.mhr70 and could not find the sibling "
                f"sam3d-body-rerun source file at {source_path}"
            ) from import_error

        spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location("_sam3d_body_mhr70", source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load SAM-3D-Body MHR70 metadata from {source_path}") from import_error

        module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_mhr70: ModuleType = _load_mhr70_module()

MHR70_ID2NAME: Final[dict[int, str]] = cast(dict[int, str], _mhr70.MHR70_ID2NAME)
MHR70_IDS: Final[list[int]] = cast(list[int], _mhr70.MHR70_IDS)
MHR70_LINKS: Final[list[tuple[int, int]]] = cast(list[tuple[int, int]], _mhr70.MHR70_LINKS)
MHR70_KEYPOINT_COUNT: Final[int] = len(MHR70_IDS)

__all__: list[str] = ["MHR70_ID2NAME", "MHR70_IDS", "MHR70_LINKS", "MHR70_KEYPOINT_COUNT"]
