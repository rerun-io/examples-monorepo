import tyro

from vistadream.api.multi_image_pipeline import VGGTInferenceConfig, run_inference

if __name__ == "__main__":
    run_inference(tyro.cli(VGGTInferenceConfig))
