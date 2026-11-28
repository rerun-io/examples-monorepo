import tyro

from fdanyone_rerun.apis.birefnet_engine import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
