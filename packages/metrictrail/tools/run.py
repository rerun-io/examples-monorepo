import tyro

from metrictrail.apis.run import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
