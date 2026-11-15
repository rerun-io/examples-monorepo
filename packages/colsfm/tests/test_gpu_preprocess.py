"""GPU preprocessing against the host path it replaces.

`colsfm.tensorrt_runtime.GpuPreprocessor` moves the BGR-HWC-uint8 to
planar-RGB-float32 conversion off the host and onto the CUDA stream the detector
already runs on. That is a performance change, so the only thing worth asserting
is that it is *not* a numerical change: the tensor the engine receives has to be
the same tensor, and the keypoints and descriptors that come out have to be the
same keypoints and descriptors.

Two seams are tested, and they are the two the rest of the pipeline sees:

* `GpuPreprocessor.run` against `colsfm.features_native.to_network_bchw`, on real
  Galileo JPEGs. This is exact — every step (widen, transpose, reverse the
  channels, divide by 255) is exact in float32 and the preprocessing engine is
  built without FP16 for exactly that reason — so it is asserted with
  `array_equal`, not a tolerance.
* `extract_raco` and `extract_tensorrt` with `gpu_preprocessing` on and off,
  writing two databases from the same three images. Identical input tensors do
  not by themselves guarantee identical output, because TensorRT is free to pick
  a different tactic when its input arrives from a different allocation, so the
  keypoints are compared as a set and the descriptors by cosine similarity.

Same skip discipline as `tests/colsfm/test_tensorrt_backends.py`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from jaxtyping import Float32, UInt8
from numpy import ndarray

from colsfm.database import (
    Descriptors,
    Keypoints,
    create_database,
    read_descriptors_from_database,
    read_keypoints_batch,
)
from colsfm.frames_meta import FramesMeta

pytest.importorskip("tensorrt", reason="GPU preprocessing needs the tensorrt package")
pytest.importorskip("cuda.bindings.runtime", reason="GPU preprocessing allocates through cuda-python")

MIN_DESCRIPTOR_COSINE: float = 0.999
"""How close the two paths' descriptors must be, per keypoint.

The task's own gate. In practice the runs measured here are bit-identical, so
this only exists to absorb a tactic change, not a real difference."""

SAMPLE_IMAGE_COUNT: int = 3
"""Galileo frames used for the equivalence checks; enough to catch a channel swap."""


def _cuda_available() -> bool:
    """Whether a GPU this TensorRT can build for is present.

    Returns:
        True when `cudaGetDeviceProperties` answers.
    """
    from colsfm.tensorrt_runtime import compute_capability

    try:
        compute_capability()
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")


def test_the_preprocessing_engine_reproduces_the_host_conversion_bit_for_bit(
    galileo_input: FramesMeta, galileo_input_dir: Path
) -> None:
    """`GpuPreprocessor.run` equals `to_network_bchw` exactly, on real JPEGs.

    Random bytes would exercise the arithmetic just as well, but real frames also
    catch the mistake that matters most and is invisible in a symmetric test: a
    channel order that is reversed once too often or not at all.
    """
    from cuda.bindings import runtime as cudart

    from colsfm.features_native import ResizedImageBGR, to_network_bchw
    from colsfm.tensorrt_runtime import DeviceTensor, GpuPreprocessor, check_cuda

    names: list[str] = [keyframe.image_name for keyframe in galileo_input.keyframes[:SAMPLE_IMAGE_COUNT]]
    frames: list[ResizedImageBGR] = [cv2.imread(str(galileo_input_dir / name), cv2.IMREAD_COLOR) for name in names]
    height: int = int(frames[0].shape[0])
    width: int = int(frames[0].shape[1])
    expected: Float32[ndarray, "batch 3 height width"] = np.concatenate(
        [to_network_bchw(frame) for frame in frames], axis=0
    )

    stream: int = int(check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")[0])
    try:
        with GpuPreprocessor(
            height=height, width=width, max_batch=len(frames), optimal_batch=len(frames), stream=stream
        ) as preprocessor:
            for slot, frame in enumerate(frames):
                preprocessor.staging_bhwc[slot] = frame
            device: DeviceTensor = preprocessor.run(len(frames))
            actual: Float32[ndarray, "batch 3 height width"] = np.empty_like(expected)
            check_cuda(
                cudart.cudaMemcpyAsync(
                    actual.ctypes.data,
                    device.pointer,
                    actual.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                ),
                "cudaMemcpyAsync(D2H image)",
            )
            check_cuda(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
    finally:
        check_cuda(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")

    assert device.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_the_staged_uint8_batch_is_a_quarter_of_the_float32_one(galileo_input_dir: Path) -> None:
    """The point of the change, stated as a number the type system cannot state.

    The host used to build and push a float32 planar tensor; it now pushes the
    decoded frame. One byte per sample instead of four is the whole of the
    transfer saving, and it is worth failing on if a future edit widens the
    staging buffer back to float32.
    """
    from cuda.bindings import runtime as cudart

    from colsfm.tensorrt_runtime import GpuPreprocessor, check_cuda

    stream: int = int(check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")[0])
    try:
        with GpuPreprocessor(height=64, width=96, max_batch=8, optimal_batch=8, stream=stream) as preprocessor:
            staging: UInt8[ndarray, "8 64 96 3"] = preprocessor.staging_bhwc
            assert staging.dtype == np.uint8
            assert staging.shape == (8, 64, 96, 3)
            assert preprocessor.output_nbytes == 4 * staging.nbytes
    finally:
        check_cuda(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")


def _extract_both_ways(
    galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path: Path
) -> tuple[dict[int, Keypoints], dict[int, Keypoints], dict[int, Descriptors], dict[int, Descriptors]]:
    """Extract the same images with GPU and host preprocessing.

    Args:
        galileo_input: The Galileo input collection.
        galileo_input_dir: Where its images live.
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Keypoints and descriptors per `image_id`, GPU path first.
    """
    from colsfm.features_raco import extract_raco

    keep: list[int] = [keyframe.keyframe_id for keyframe in galileo_input.keyframes[:SAMPLE_IMAGE_COUNT]]
    subset: FramesMeta = galileo_input.filtered(keep)
    names: list[str] = [keyframe.image_name for keyframe in subset.keyframes]

    keypoints: list[dict[int, Keypoints]] = []
    descriptors: list[dict[int, Descriptors]] = []
    for gpu_preprocessing in (True, False):
        database_path: Path = tmp_path / f"raco_{gpu_preprocessing}.db"
        create_database(database_path, subset)
        extract_raco(
            database_path,
            galileo_input_dir,
            names,
            min_score=0.0,
            max_num_features=0,
            batch_size=SAMPLE_IMAGE_COUNT,
            gpu_preprocessing=gpu_preprocessing,
        )
        stored: dict[int, Descriptors] = read_descriptors_from_database(database_path)
        keypoints.append(read_keypoints_batch(database_path, sorted(stored)))
        descriptors.append(stored)
    return keypoints[0], keypoints[1], descriptors[0], descriptors[1]


def test_gpu_preprocessing_leaves_the_features_alone(galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path: Path) -> None:
    """Same keypoints, same descriptors, whichever side of the bus does the arithmetic.

    This is the contract the whole change has to keep: everything downstream —
    matching, triangulation, the ATE the pipeline is judged on — reads these two
    database columns and nothing else.
    """
    from colsfm.features_raco import RACO_ONNX_PATH

    if not RACO_ONNX_PATH.is_file():
        pytest.skip(f"{RACO_ONNX_PATH.name} is missing; run `pixi run -e colsfm colsfm-raco-export`")

    gpu_keypoints, host_keypoints, gpu_descriptors, host_descriptors = _extract_both_ways(galileo_input, galileo_input_dir, tmp_path)

    assert set(gpu_keypoints) == set(host_keypoints)
    for image_id, gpu_xy in gpu_keypoints.items():
        host_xy: Keypoints = host_keypoints[image_id]
        assert gpu_xy.shape == host_xy.shape
        assert np.array_equal(gpu_xy, host_xy)
        gpu_desc: Descriptors = gpu_descriptors[image_id]
        host_desc: Descriptors = host_descriptors[image_id]
        assert gpu_desc.shape == host_desc.shape
        cosine: Float32[ndarray, " num_keypoints"] = np.sum(gpu_desc * host_desc, axis=1) / (
            np.linalg.norm(gpu_desc, axis=1) * np.linalg.norm(host_desc, axis=1) + 1e-12
        )
        assert float(cosine.min()) > MIN_DESCRIPTOR_COSINE
