# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class DeferredBuildExtension(_build_ext):
    """Deferred CUDA build — only imports torch when actually compiling."""

    def build_extensions(self):
        from torch import cuda
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()

        ext = CUDAExtension(
            name="curope",
            sources=["curope.cpp", "kernels.cu"],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                cxx=["-O3"],
            ),
        )

        self.extensions = [ext]
        torch_builder = BuildExtension(self.distribution)
        torch_builder.extensions = self.extensions
        torch_builder.build_temp = self.build_temp
        torch_builder.build_lib = self.build_lib
        torch_builder.inplace = self.inplace
        torch_builder.build_extensions()


setup(
    name="curope",
    ext_modules=[Extension("curope", sources=[])],
    cmdclass={"build_ext": DeferredBuildExtension},
)
