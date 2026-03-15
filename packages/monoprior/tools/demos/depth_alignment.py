import tyro

from monopriors.apis.depth_alignment import DepthAlignmentCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(DepthAlignmentCLIConfig))
