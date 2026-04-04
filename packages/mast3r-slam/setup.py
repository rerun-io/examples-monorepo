from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Use CONDA_PREFIX for eigen headers (set by pixi environment activation)
conda_prefix = os.environ.get("CONDA_PREFIX", "")
eigen_include = os.path.join(conda_prefix, "include", "eigen3")

include_dirs = [
    os.path.join(ROOT, "mast3r_slam/backend/include"),
    eigen_include,
]

sources = [
    "mast3r_slam/backend/src/gn.cpp",
]
extra_compile_args = {
    "cores": ["j8"],
    "cxx": ["-O3"],
}

has_cuda = torch.cuda.is_available()
if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension

    sources.append("mast3r_slam/backend/src/gn_kernels.cu")
    sources.append("mast3r_slam/backend/src/matching_kernels.cu")
    extra_compile_args["nvcc"] = [
        "-O3",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_120,code=sm_120",
    ]
    ext_modules = [
        CUDAExtension(
            "mast3r_slam_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    raise RuntimeError("CUDA not found, cannot compile mast3r_slam_backends!")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
