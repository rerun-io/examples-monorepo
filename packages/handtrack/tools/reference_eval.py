"""UmeTrack reference ladder CLI."""
import tyro

from handtrack.apis.reference_eval import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
