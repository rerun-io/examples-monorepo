"""CLI entrypoint for ARKitScenes PromptDA catalog inference."""

import tyro

from rerun_prompt_da.apis.prompt_da_arkitscenes import PDAArkitScenesConfig, main

if __name__ == "__main__":
    main(tyro.cli(PDAArkitScenesConfig))
