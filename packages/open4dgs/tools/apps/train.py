import tyro

from open4dgs.apis.train import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
