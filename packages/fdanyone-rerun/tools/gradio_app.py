import tyro

from fdanyone_rerun.apis.app import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
