import tyro

from posekit.hands_ui import Config, launch

if __name__ == "__main__":
    launch(tyro.cli(Config))
