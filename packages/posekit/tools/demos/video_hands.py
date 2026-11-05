import tyro

from posekit.apis.video_hands import VideoHandsConfig, main

if __name__ == "__main__":
    main(tyro.cli(VideoHandsConfig))
