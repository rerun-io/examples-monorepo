import tyro

from metrictrail.apis.check_kernels import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
