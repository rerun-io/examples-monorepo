import tyro

from monopriors.apis.multiview_calibration import MVInferenceConfig, main

if __name__ == "__main__":
    main(tyro.cli(MVInferenceConfig))
