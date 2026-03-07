import tyro

from monopriors.apis.metric_depth_inference import MetricDepthConfig, metric_depth_from_img

if __name__ == "__main__":
    metric_depth_from_img(tyro.cli(MetricDepthConfig))
