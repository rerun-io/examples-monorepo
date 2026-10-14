import tyro

from mast3r_slam.apis.slam import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config, description="Dense monocular SLAM (MASt3R-SLAM) of a catalog segment, a video, an image directory or a webcam."))
