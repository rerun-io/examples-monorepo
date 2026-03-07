import tyro

from monopriors.apis.multiview_inference import VGGTInferenceConfig, run_inference

if __name__ == "__main__":
    run_inference(tyro.cli(VGGTInferenceConfig))
