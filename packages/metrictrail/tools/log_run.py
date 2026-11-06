import tyro

from metrictrail.apis.log_run import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
