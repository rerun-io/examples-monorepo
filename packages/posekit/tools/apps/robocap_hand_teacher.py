import tyro

from posekit.apis.robocap_hand_teacher import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
