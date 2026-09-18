"""Tyro entrypoint for one-session conversion."""

import tyro

from agent_traces.apis.convert import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
