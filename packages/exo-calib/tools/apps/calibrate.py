import tyro

from exo_calib.apis.calibrate import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config, description="Calibrate the exo cameras of one catalog segment (Stage A once, two Stage B + refinement passes, report)."))
