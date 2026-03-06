import tyro

from monopriors.apis.relative_depth_inference import PredictorConfig, relative_depth_from_img

if __name__ == "__main__":
    relative_depth_from_img(tyro.cli(PredictorConfig))
