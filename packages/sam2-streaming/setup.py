# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from setuptools import setup

# The sam2._C connected-components kernel is broken against torch 2.12: it
# hits an internal assert on valid CUDA tensors, then sam2 silently falls back.
# Outputs are byte-identical without it. Set SAM2_BUILD_CUDA=1 after fixing the
# kernel to re-enable the extension.
BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "0") == "1"
# By default, we allow SAM 2 installation to proceed even with build errors.
# You may force stopping on errors with `export SAM2_BUILD_ALLOW_ERRORS=0`.
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

CUDA_HOME = os.getenv("CUDA_HOME", None)
if CUDA_HOME is None and BUILD_CUDA and not BUILD_ALLOW_ERRORS:
    raise RuntimeError(
        "BUILD_CUDA is set to 1, but CUDA_HOME is not set. "
        "Please set CUDA_HOME to the path of your CUDA installation."
    )

# Catch and skip errors during extension building and print a warning message
# (note that this message only shows up under verbose build mode
# "pip install -v -e ." or "python setup.py build_ext -v")
CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the SAM 2 CUDA extension due to the error above. "
    "You can still use SAM 2 and it's OK to ignore the error above, although some "
    "post-processing functionality may be limited (which doesn't affect the results in most cases; "
    "(see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).\n"
)


def get_extensions():
    if not BUILD_CUDA:
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension

        srcs = ["sam2/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            ext_modules = []
        else:
            raise e

    return ext_modules


cmdclass = {}
if BUILD_CUDA:
    try:
        from torch.utils.cpp_extension import BuildExtension

        class BuildExtensionIgnoreErrors(BuildExtension):

            def finalize_options(self):
                try:
                    super().finalize_options()
                except Exception as e:
                    print(CUDA_ERROR_MSG.format(e))
                    self.extensions = []

            def build_extensions(self):
                try:
                    super().build_extensions()
                except Exception as e:
                    print(CUDA_ERROR_MSG.format(e))
                    self.extensions = []

            def get_ext_filename(self, ext_name):
                try:
                    return super().get_ext_filename(ext_name)
                except Exception as e:
                    print(CUDA_ERROR_MSG.format(e))
                    self.extensions = []
                    return "_C.so"

        cmdclass = {
            "build_ext": (
                BuildExtensionIgnoreErrors.with_options(no_python_abi_suffix=True)
                if BUILD_ALLOW_ERRORS
                else BuildExtension.with_options(no_python_abi_suffix=True)
            )
        }
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
        else:
            raise e
# Setup configuration
setup(
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
)
