import tyro

from pysfm.apis.pycolmap_vid_recon import VidReconCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(VidReconCLIConfig))
