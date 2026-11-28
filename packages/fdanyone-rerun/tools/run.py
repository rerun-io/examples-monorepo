import tyro

from fdanyone_rerun.apis.run import Config, exit_now, main

if __name__ == "__main__":
    config: Config = tyro.cli(Config)
    main(config)
    exit_now(config.rr_config.rec_stream)
