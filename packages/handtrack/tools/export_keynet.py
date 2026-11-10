import tyro

from handtrack.apis.export_keynet import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
