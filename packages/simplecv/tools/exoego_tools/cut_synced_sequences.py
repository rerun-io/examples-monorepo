import tyro

from simplecv.apis.exoego_tools.cut_synced_sequences import Config, main

if __name__ == "__main__":
    config: Config = tyro.cli(Config)
    main(config)
