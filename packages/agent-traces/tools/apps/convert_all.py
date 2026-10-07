"""Convert selected sessions from a Claude home."""

import tyro

from agent_traces.apis.convert_all import Config, main

if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Config)).exit_code)
