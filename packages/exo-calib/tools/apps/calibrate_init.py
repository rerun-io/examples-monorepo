import tyro

from exo_calib.apis.calibrate_init import InitCalibrationConfig, main

if __name__ == "__main__":
    main(tyro.cli(InitCalibrationConfig, description="Stage A: G3T + MoGe-2 metric initial exo-camera calibration from the catalog."))
