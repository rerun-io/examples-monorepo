import tyro

from mv_api.api.metric_exoego_calib import RRDPipelineConfig, main

if __name__ == "__main__":
    main(tyro.cli(RRDPipelineConfig))
