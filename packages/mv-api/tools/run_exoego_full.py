import tyro

from mv_api.api.full_exoego_pipeline import RRDPipelineConfig, main

if __name__ == "__main__":
    main(tyro.cli(RRDPipelineConfig))
