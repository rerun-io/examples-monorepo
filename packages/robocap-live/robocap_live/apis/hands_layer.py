"""Find the ONNX Runtime library for the native hands binding."""

import importlib.util
import os
from pathlib import Path

def default_ort_dylib() -> Path:
    """``ORT_DYLIB_PATH``, else the ``libonnxruntime.so`` of the env's ``onnxruntime`` package, found without importing it.

    Importing the package would load its own copy of ONNX Runtime beside the one the core loads.

    Raises:
        ValueError: If neither exists.
    """
    from_env: str | None = os.environ.get("ORT_DYLIB_PATH")
    if from_env:
        return Path(from_env)
    spec = importlib.util.find_spec("onnxruntime")
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            libraries: list[Path] = sorted((Path(location) / "capi").glob("libonnxruntime.so*"))
            if libraries:
                return libraries[-1]
    raise ValueError("no ONNX Runtime library: pass ort_dylib or set ORT_DYLIB_PATH (or run in the robocap-live env)")

