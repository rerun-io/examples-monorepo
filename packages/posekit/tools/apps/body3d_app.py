import tyro

from posekit.body3d_ui import Config, launch

if __name__ == "__main__":
    launch(tyro.cli(Config))
