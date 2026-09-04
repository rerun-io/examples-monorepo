"""Tyro shim for the ultrawide PromptDA-L reference ceiling."""

import tyro

from zipdepth.apis.eval_teacher_reference import TeacherReferenceConfig, main

if __name__ == "__main__":
    main(tyro.cli(TeacherReferenceConfig))
