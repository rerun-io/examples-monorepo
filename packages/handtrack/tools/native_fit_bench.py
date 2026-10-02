"""Thin CLI: torch vs native fit on the recorded golden calls (``handtrack.apis.native_fit_bench``); run in the handtrack env."""

import tyro

from handtrack.apis.native_fit_bench import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
