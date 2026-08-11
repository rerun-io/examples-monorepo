import tyro

from mvs.apis.live_mesh import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
