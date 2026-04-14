"""Build script for dpvo native CUDA extensions.

When installed via Pixi, dpvo is treated as a pure-Python package.
The package's custom CUDA kernels are built explicitly with:

    python setup.py build_ext --inplace

inside the activated Pixi environment, while `lietorch` comes from a
prebuilt conda package.
"""

import os
import sys

from setuptools import setup

if "build_ext" in sys.argv:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found, cannot compile dpvo native extensions")

    root = os.path.dirname(os.path.abspath(__file__))
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}.{cap[1]}+PTX"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        print(f"Auto-detected CUDA arch: {arch}")

    setup(
        ext_modules=[
            CUDAExtension(
                "dpvo._cuda_corr",
                sources=[
                    "dpvo/altcorr/correlation.cpp",
                    "dpvo/altcorr/correlation_kernel.cu",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            ),
            CUDAExtension(
                "dpvo._cuda_ba",
                sources=["dpvo/fastba/ba.cpp", "dpvo/fastba/ba_cuda.cu", "dpvo/fastba/block_e.cu"],
                include_dirs=[os.path.join(conda_prefix, "include", "eigen3")],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
else:
    setup()
