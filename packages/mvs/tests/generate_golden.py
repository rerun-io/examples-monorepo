"""Generate deterministic CPU ONNX outputs for depth-runtime parity tests."""

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import tyro
from jaxtyping import Float32, UInt8
from numpy import ndarray

SEED: int = 20260810
DEFAULT_OUTPUT: Path = Path(__file__).parent / "data" / "depth_golden.npz"


def seeded_inputs(seed: int, batch_size: int = 1) -> dict[str, ndarray]:
    """Create deterministic normalized images and calibrated camera geometry.

    Args:
        seed: NumPy random seed for the uint8 RGB inputs.
        batch_size: Number of distinct depth queries to create.

    Returns:
        Float32 arrays keyed by the ONNX model's six input names.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    cur_rgb_b3hw: UInt8[ndarray, "b 3 384 512"] = rng.integers(0, 256, size=(batch_size, 3, 384, 512), dtype=np.uint8)
    src_rgb_bm3hw: UInt8[ndarray, "b 7 3 384 512"] = np.empty((batch_size, 7, 3, 384, 512), dtype=np.uint8)
    batch_index: int
    source_index: int
    for batch_index in range(batch_size):
        for source_index in range(7):
            src_rgb_bm3hw[batch_index, source_index] = np.roll(
                cur_rgb_b3hw[batch_index], shift=-(source_index - 3) * 4, axis=2
            )
    mean_11311: Float32[ndarray, "1 1 3 1 1"] = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :, None, None]
    std_11311: Float32[ndarray, "1 1 3 1 1"] = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :, None, None]
    cur_image_b3hw: Float32[ndarray, "b 3 384 512"] = (cur_rgb_b3hw.astype(np.float32) / 255.0 - mean_11311[:, 0]) / std_11311[:, 0]
    src_image_bm3hw: Float32[ndarray, "b 7 3 384 512"] = (src_rgb_bm3hw.astype(np.float32) / 255.0 - mean_11311) / std_11311

    K_s1_44: Float32[ndarray, "4 4"] = np.array(
        [[128.0, 0.0, 64.0, 0.0], [0.0, 128.0, 48.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    src_K_bm44: Float32[ndarray, "b 7 4 4"] = np.repeat(np.repeat(K_s1_44[None, None], batch_size, axis=0), 7, axis=1)
    cur_invK_b44: Float32[ndarray, "b 4 4"] = np.repeat(
        np.linalg.inv(K_s1_44.astype(np.float64)).astype(np.float32)[None], batch_size, axis=0
    )

    src_world_T_cam_m44: Float32[ndarray, "7 4 4"] = np.repeat(np.eye(4, dtype=np.float32)[None], 7, axis=0)
    for source_index in range(7):
        yaw: float = (source_index - 3) * 0.002
        cosine: float = float(np.cos(yaw))
        sine: float = float(np.sin(yaw))
        src_world_T_cam_m44[source_index, :3, :3] = np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=np.float32
        )
        src_world_T_cam_m44[source_index, :3, 3] = np.array(
            [(source_index - 3) * 0.05, 0.0, 0.0], dtype=np.float32
        )
    src_cam_T_world_bm44: Float32[ndarray, "b 7 4 4"] = np.repeat(
        np.linalg.inv(src_world_T_cam_m44.astype(np.float64)).astype(np.float32)[None], batch_size, axis=0
    )
    cur_world_T_cam_b44: Float32[ndarray, "b 4 4"] = np.repeat(np.eye(4, dtype=np.float32)[None], batch_size, axis=0)

    return {
        "cur_image_b3hw": cur_image_b3hw,
        "src_image_bm3hw": src_image_bm3hw,
        "src_K_bm44": src_K_bm44,
        "cur_invK_b44": cur_invK_b44,
        "src_cam_T_world_bm44": src_cam_T_world_bm44,
        "cur_world_T_cam_b44": cur_world_T_cam_b44,
    }


def main(model_path: Path, output_path: Path = DEFAULT_OUTPUT) -> None:
    """Run the model on CPU and write the deterministic reference outputs.

    Args:
        model_path: Source ONNX model.
        output_path: Destination compressed NumPy archive.
    """
    session: Any = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs: dict[str, ndarray] = seeded_inputs(SEED)
    outputs: list[ndarray] = session.run(["depth_pred_s0_b1hw", "lowest_cost_bhw"], inputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, seed=np.array(SEED, dtype=np.int64), depth_pred_s0_b1hw=outputs[0], lowest_cost_bhw=outputs[1])
    print(f"wrote {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
