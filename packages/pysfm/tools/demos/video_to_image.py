import tyro

from pysfm.apis.video_to_image import VideoToImageCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(VideoToImageCLIConfig))
