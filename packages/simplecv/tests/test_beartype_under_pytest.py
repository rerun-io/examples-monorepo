"""Beartype checks simplecv while pytest runs (pytest-beartype, `beartype_packages` in pyproject.toml)."""

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from simplecv.camera_orient_utils import rotation_matrix_between


def test_annotation_violation_raises_under_pytest() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        rotation_matrix_between(np.zeros(4), np.zeros(3))
