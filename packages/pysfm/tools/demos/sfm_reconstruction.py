import tyro

from pysfm.apis.sfm_reconstruction import SfMCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(SfMCLIConfig))
