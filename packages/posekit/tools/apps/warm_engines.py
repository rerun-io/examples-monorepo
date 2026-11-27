import tyro

from posekit.apis.warm_engines import WarmEnginesConfig, main

if __name__ == "__main__":
    main(tyro.cli(WarmEnginesConfig))
