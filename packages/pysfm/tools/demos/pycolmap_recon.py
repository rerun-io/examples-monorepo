import tyro

from pysfm.apis.pycolmap_recon import RigReconCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(RigReconCLIConfig))
