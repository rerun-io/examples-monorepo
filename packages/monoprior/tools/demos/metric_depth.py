import tyro

from monopriors.apis.metric_depth import MetricDepthCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(MetricDepthCLIConfig))
