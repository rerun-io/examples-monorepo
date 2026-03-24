import tyro

from pysfm.apis.pycolmap_recon import SfMCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(SfMCLIConfig))
