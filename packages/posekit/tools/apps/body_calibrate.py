import tyro

from posekit.apis.body_calibrate import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
