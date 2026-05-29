import tyro

from simplecv.apis.metric_exoego_calib import RRDPipelineConfig, main

if __name__ == "__main__":
    main(tyro.cli(RRDPipelineConfig))
