"""Build script for mast3r_slam._backends CUDA extension.

When installed as a pypi-dependency (via pixi), this is a pure-Python package.
The CUDA extension is compiled separately via a pixi task:

    python setup.py build_ext --inplace

inside the activated pixi environment (which provides torch + nvcc).

The extension is namespaced inside the package (``mast3r_slam._backends``)
so that ``--inplace`` places the ``.so`` inside ``mast3r_slam/``, where the
editable install's finder already looks.  This follows the same pattern used
by PyTorch3D (``pytorch3d._C``) and Detectron2 (``detectron2._C``).
"""

import os
import sys

from setuptools import setup

if "build_ext" in sys.argv:
    # Only import torch when explicitly building the CUDA extension
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found, cannot compile mast3r_slam._backends!")

    ROOT = os.path.dirname(os.path.abspath(__file__))
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    # Auto-detect GPU architecture if TORCH_CUDA_ARCH_LIST isn't set.
    # CUDAExtension reads this env var to generate the correct -gencode flags.
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}.{cap[1]}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        print(f"Auto-detected CUDA arch: {arch}")

    setup(
        ext_modules=[
            CUDAExtension(
                "mast3r_slam._backends",
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
                    "nvcc": ["-O3"],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
else:
    # Pure-Python install for pixi's pypi resolver
    setup()
