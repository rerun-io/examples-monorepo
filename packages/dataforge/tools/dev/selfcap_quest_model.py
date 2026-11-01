"""Rebuild selfcap_devices/quest3s.rrd from the Quest 3S model package's quest_3s.glb."""

import tyro

from dataforge.datasets.selfcap import build_quest_model

if __name__ == "__main__":
    tyro.cli(build_quest_model)
