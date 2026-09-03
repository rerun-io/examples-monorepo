"""CLI shim for fixed hard-frame evaluation."""

import tyro

from zipdepth.apis.eval_hard_frames import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
