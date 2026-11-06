# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# CroCo submodule import (or package import)
# --------------------------------------------------------
#
# Replacement file: copied over upstream's path_to_croco.py during build.
# Same pattern as fix-path-to-dust3r.patch — try importing the installed
# package first. If CroCo's `models` package is installed (our case via
# pixi-build), no sys.path hack is needed. Falls back to the submodule
# layout for standalone upstream usage.

import sys
import os.path as path

# If croco models are already installed as a package, no path hack needed.
try:
    import models.croco  # noqa: F401
except ImportError:
    # Fall back to submodule path layout.
    HERE_PATH = path.normpath(path.dirname(__file__))
    CROCO_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../croco'))
    CROCO_MODELS_PATH = path.join(CROCO_REPO_PATH, 'models')
    if path.isdir(CROCO_MODELS_PATH):
        sys.path.insert(0, CROCO_REPO_PATH)
    else:
        raise ImportError(
            "croco is not installed and not found as a submodule. "
            "Install as a package or use git submodule update --init --recursive."
        )
