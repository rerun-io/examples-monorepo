import tyro

from posekit.apis.fullcircle_masks import FullCircleMasksConfig, main

if __name__ == "__main__":
    main(tyro.cli(FullCircleMasksConfig))
