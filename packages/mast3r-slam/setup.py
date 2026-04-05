"""Build script for mast3r_slam_backends CUDA extension.

When installed as a pypi-dependency (via pixi), this is a pure-Python package.
The CUDA extension is compiled separately via a pixi task:

    python setup.py build_ext --inplace

inside the activated pixi environment (which provides torch + nvcc).
"""

import os
import sys

from setuptools import setup

if "build_ext" in sys.argv:
    # Only import torch when explicitly building the CUDA extension
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found, cannot compile mast3r_slam_backends!")

    ROOT = os.path.dirname(os.path.abspath(__file__))
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    setup(
        ext_modules=[
            CUDAExtension(
                "mast3r_slam_backends",
                include_dirs=[
                    os.path.join(ROOT, "mast3r_slam/backend/include"),
                    os.path.join(conda_prefix, "include", "eigen3"),
                ],
                sources=[
                    "mast3r_slam/backend/src/gn.cpp",
                    "mast3r_slam/backend/src/gn_kernels.cu",
                    "mast3r_slam/backend/src/matching_kernels.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": [
                        "-O3",
                        "-gencode=arch=compute_75,code=sm_75",
                        "-gencode=arch=compute_80,code=sm_80",
                        "-gencode=arch=compute_86,code=sm_86",
                        "-gencode=arch=compute_89,code=sm_89",
                        "-gencode=arch=compute_90,code=sm_90",
                        "-gencode=arch=compute_120,code=sm_120",
                    ],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
else:
    # Pure-Python install for pixi's pypi resolver
    setup()
