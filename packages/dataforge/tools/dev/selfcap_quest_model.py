"""Rebuild selfcap_devices/quest3s.rrd from the Quest 3S model package's quest_3s.glb.

Usage: python tools/dev/selfcap_quest_model.py <quest_3s.glb>
"""

import sys
from pathlib import Path

from dataforge.datasets.selfcap import DEVICE_MESH_FILES, DEVICE_MESHES, build_quest_model

if __name__ == "__main__":
    build_quest_model(Path(sys.argv[1]), DEVICE_MESHES / f"{DEVICE_MESH_FILES['quest']}.rrd")
