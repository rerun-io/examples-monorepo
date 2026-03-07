import tyro

from monopriors.apis.surface_normal_inference import NormalPredictorConfig, surface_normal_from_img

if __name__ == "__main__":
    surface_normal_from_img(tyro.cli(NormalPredictorConfig))
