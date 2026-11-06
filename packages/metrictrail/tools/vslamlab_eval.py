import tyro

from metrictrail.apis.vslamlab_eval import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
