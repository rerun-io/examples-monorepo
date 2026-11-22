import tyro

from fdanyone_rerun.apis.register_catalog import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
