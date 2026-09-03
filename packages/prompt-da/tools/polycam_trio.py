"""Thin CLI shim for the three-way ARKit / PromptDA / ZipDepth-PromptDA Polycam comparison."""

import tyro

from rerun_prompt_da.apis.polycam_trio import PolycamTrioConfig, main

if __name__ == "__main__":
    main(tyro.cli(PolycamTrioConfig))
