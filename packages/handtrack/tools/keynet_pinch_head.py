import tyro

from handtrack.apis.keynet_pinch_head import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
