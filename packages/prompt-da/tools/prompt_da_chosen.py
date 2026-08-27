"""CLI entrypoint for chosen-frame ARKitScenes PromptDA inference."""

import tyro

from rerun_prompt_da.apis.prompt_da_chosen import PDAChosenConfig, main

if __name__ == "__main__":
    main(tyro.cli(PDAChosenConfig))
