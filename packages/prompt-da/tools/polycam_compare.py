"""Thin CLI shim for the PromptDA-versus-ZipDepth-PromptDA Polycam comparison."""

import tyro

from rerun_prompt_da.apis.polycam_compare import PolycamCompareConfig, main

if __name__ == "__main__":
    main(tyro.cli(PolycamCompareConfig))
