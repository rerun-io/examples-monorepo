import tyro

from depth_video_viewer.apis.comparison import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
