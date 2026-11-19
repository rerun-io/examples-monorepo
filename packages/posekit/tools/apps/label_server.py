import tyro

from posekit.apis.label_server import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
