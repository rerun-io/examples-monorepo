"""CLI entrypoint for the PromptDA Polycam demo."""

import tyro

from rerun_prompt_da.apis.prompt_da_polycam import PDAPolycamConfig, pda_polycam_inference

if __name__ == "__main__":
    """Parse CLI arguments with Tyro and run the PromptDA pipeline."""

    pda_polycam_inference(tyro.cli(PDAPolycamConfig))
