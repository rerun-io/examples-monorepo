import tyro

from posekit.apis.video_track import VideoTrackConfig, main

if __name__ == "__main__":
    main(tyro.cli(VideoTrackConfig))
