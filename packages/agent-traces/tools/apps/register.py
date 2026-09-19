"""Register converted sessions in a catalog."""

import tyro

from agent_traces.apis.register import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
