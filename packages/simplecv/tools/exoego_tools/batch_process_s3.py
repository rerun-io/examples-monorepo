import tyro

from simplecv.apis.exoego_tools.batch_process_s3 import Config, main

if __name__ == "__main__":
    config: Config = tyro.cli(Config)
    main(config)
