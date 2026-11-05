import tyro

from posekit.apis.sapiens_calibrate import SapiensCalibrateConfig, main

if __name__ == "__main__":
    main(tyro.cli(SapiensCalibrateConfig))
