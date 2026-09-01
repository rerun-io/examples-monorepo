import tyro

from exo_calib.apis.keypoints2d import Keypoints2dConfig, main

if __name__ == "__main__":
    main(tyro.cli(Keypoints2dConfig, description="Stage B: YOLOX + Sapiens2 COCO-133 keypoints over the exo videos, from the catalog."))
