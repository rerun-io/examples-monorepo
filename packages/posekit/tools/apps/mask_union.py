import tyro

from posekit.apis.mask_union import MaskUnionConfig, main

if __name__ == "__main__":
    main(tyro.cli(MaskUnionConfig))
