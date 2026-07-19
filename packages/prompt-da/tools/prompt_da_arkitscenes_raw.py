import tyro

from rerun_prompt_da.apis.prompt_da_arkitscenes_raw import PDAArkitScenesRawConfig, main

if __name__ == "__main__":
    main(tyro.cli(PDAArkitScenesRawConfig))
