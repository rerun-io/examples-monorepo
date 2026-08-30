import tyro

from monopriors.apis.stereo_depth import StereoDepthCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(StereoDepthCLIConfig))
