"""Write the golden set's cold rows as a plain-text fixture for the Rust ``cold_bench`` example (numpy only).

The fixture lets the cold fit (``handfit::cold::initial_pose``) be timed and checked without Python, on any host the Rust
example cross-builds for (the RoboCap cap). Every number is the exact float64 value of the recorded float32 input, so the example
builds bit-identical inputs to ``handfit.HandFitter``. Format, one record per line, numbers separated by spaces:

- ``config <13 numbers>``: the recorded FitConfig in ``HandFitter`` order, without phi.
- ``model <id> <axes 60> <pivots 60> <rest 63> <weights 63> <limits 40>``: row-major, the first 20 joints only.
- ``row <npz row> <model id> <side> <phi> <views>``, then per view
  ``view <cam_from_rig 16> <world_from_rig 16> <focal 2> <principal 2> <fisheye 0|1> <fisheye62 8> <pixels 42> <weights 21> <d_rel 21>``,
  then ``torch <rotation 9> <translation 3> <joint angles 22> <energy> <converged 0|1>`` (the recorded torch result).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray

from handfit.apis.golden_bench import CONFIG_FIELDS


@dataclass(frozen=True, slots=True)
class Config:
    """Export the cold rows (``has_previous`` false) of the golden set for ``cargo run --example cold_bench``."""

    golden: Path
    """The npz written by handtrack's export_golden.py; its .json index (the FitConfig) sits beside it."""
    output: Path
    """Where the text fixture goes."""


def numbers(values: Float32[ndarray, "..."] | Int64[ndarray, "..."] | list[float]) -> str:
    """Space-separated shortest round-trip reprs; float32 values widen to float64 exactly."""
    return " ".join(repr(float(x)) for x in np.asarray(values).reshape(-1).tolist())


def main(config: Config) -> None:
    arrays: dict[str, ndarray] = dict(np.load(config.golden))
    fit_config: dict[str, float] = json.loads(config.golden.with_suffix(".json").read_text())["fit_config"]
    cold: Int64[ndarray, "c"] = np.flatnonzero(~arrays["has_previous"].astype(bool))
    lines: list[str] = [f"# cold rows of {config.golden}", "config " + numbers([float(fit_config[name]) for name in CONFIG_FIELDS])]
    for model_id in sorted(set(arrays["model_id"][cold].tolist())):
        parts: list[Float32[ndarray, "..."]] = [
            arrays[f"model{model_id}_{name}"][:count]
            for name, count in [
                ("joint_rotation_axes", 20),
                ("joint_rest_positions", 20),
                ("landmark_rest_positions", 21),
                ("landmark_rest_bone_weights", 21),
                ("joint_limits", 20),
            ]
        ]
        lines.append(f"model {model_id} " + " ".join(numbers(p) for p in parts))
    for row in cold.tolist():
        views: int = int(arrays["num_views"][row])
        lines.append(f"row {row} {int(arrays['model_id'][row])} {int(arrays['side'][row])} {float(arrays['phi'][row])!r} {views}")
        for view in range(views):
            fisheye: bool = bool(arrays["is_fisheye"][row, view])
            fields: list[Float32[ndarray, "..."]] = [
                arrays["cam_from_rig"][row, view],
                arrays["world_from_rig"][row, view],
                arrays["focal"][row, view],
                arrays["principal"][row, view],
                np.array([1.0 if fisheye else 0.0], dtype=np.float32),
                arrays["fisheye62"][row, view],
                arrays["keypoints_px"][row, view],
                arrays["weights"][row, view],
                arrays["d_rel_mm"][row, view],
            ]
            lines.append("view " + " ".join(numbers(f) for f in fields))
        torch: list[str] = [
            numbers(arrays["out_rotation"][row]),
            numbers(arrays["out_translation"][row]),
            numbers(arrays["out_joint_angles"][row]),
            repr(float(arrays["out_energy"][row])),
            "1" if bool(arrays["out_converged"][row]) else "0",
        ]
        lines.append("torch " + " ".join(torch))
    config.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(cold)} cold rows to {config.output}")
