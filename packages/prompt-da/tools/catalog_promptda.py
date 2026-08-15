"""Stream PromptDA inference over ARKitScenes catalog segments."""

import tyro

from rerun_prompt_da.apis.catalog_promptda import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
