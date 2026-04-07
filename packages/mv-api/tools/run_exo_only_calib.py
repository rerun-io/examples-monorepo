import tyro

from mv_api.api.exo_only_calibration import ExoOnlyCalibConfig, main

if __name__ == "__main__":
    main(tyro.cli(ExoOnlyCalibConfig))
