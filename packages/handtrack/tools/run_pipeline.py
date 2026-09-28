import tyro

from handtrack.apis.run_pipeline import RunConfig, main

if __name__ == "__main__":
    main(tyro.cli(RunConfig))
