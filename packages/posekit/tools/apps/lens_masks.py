import tyro

from posekit.apis.lens_masks import LensMasksConfig, main

if __name__ == "__main__":
    main(tyro.cli(LensMasksConfig))
