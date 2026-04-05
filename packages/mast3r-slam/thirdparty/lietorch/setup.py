import os
import sys

from setuptools import Extension, setup

ROOT = os.path.dirname(os.path.abspath(__file__))

if "build_ext" in sys.argv:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    setup(
        name="lietorch",
        version="0.3",
        packages=["lietorch"],
        ext_modules=[
            CUDAExtension(
                "lietorch_backends",
                include_dirs=[
                    os.path.join(ROOT, "lietorch/include"),
                    os.path.join(os.environ.get("CONDA_PREFIX", ""), "include", "eigen3"),
                ],
                sources=["lietorch/src/lietorch.cpp", "lietorch/src/lietorch_gpu.cu", "lietorch/src/lietorch_cpu.cpp"],
                extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
            ),
            CUDAExtension(
                "lietorch_extras",
                sources=[
                    "lietorch/extras/altcorr_kernel.cu",
                    "lietorch/extras/corr_index_kernel.cu",
                    "lietorch/extras/se3_builder.cu",
                    "lietorch/extras/se3_inplace_builder.cu",
                    "lietorch/extras/se3_solver.cu",
                    "lietorch/extras/extras.cpp",
                ],
                extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
else:
    setup(name="lietorch", version="0.3", packages=["lietorch"])
