"""Tests for the three-way Polycam comparison API."""

from typing import get_type_hints

from monopriors.models.depth_completion import PromptDAConfig, ZipDepthPromptConfig

from rerun_prompt_da.apis.polycam_trio import PolycamTrioConfig


def test_trio_config_fixes_teacher_and_student_model_families() -> None:
    """Expose backend choices without allowing the teacher and student roles to swap."""
    hints: dict[str, object] = get_type_hints(PolycamTrioConfig)

    assert hints["teacher"] is PromptDAConfig
    assert hints["student"] is ZipDepthPromptConfig
