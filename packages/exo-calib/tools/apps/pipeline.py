import tyro

from exo_calib.apis.pipeline import PipelineConfig, main

if __name__ == "__main__":
    main(tyro.cli(PipelineConfig))
