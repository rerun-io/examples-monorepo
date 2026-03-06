import tyro

from monopriors.apis.multiview_calibration import VGGTInferenceConfig, main

if __name__ == "__main__":
    main(tyro.cli(VGGTInferenceConfig))
