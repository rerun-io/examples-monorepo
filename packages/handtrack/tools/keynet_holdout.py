import tyro

from handtrack.apis.keynet_holdout import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
