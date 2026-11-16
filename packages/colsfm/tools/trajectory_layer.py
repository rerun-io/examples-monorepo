import tyro

from colsfm.apis.trajectory_layer import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
