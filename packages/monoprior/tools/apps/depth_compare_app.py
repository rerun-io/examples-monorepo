import tyro

from monopriors.apis.depth_compare_app import DepthCompareAppConfig, main

if __name__ == "__main__":
    main(tyro.cli(DepthCompareAppConfig))
