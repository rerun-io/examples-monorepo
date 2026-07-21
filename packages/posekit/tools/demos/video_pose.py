import tyro

from posekit.apis.video_pose import VideoPoseConfig, main

if __name__ == "__main__":
    main(tyro.cli(VideoPoseConfig))
