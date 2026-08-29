import tyro

from posekit.track_ui import AppConfig, launch

if __name__ == "__main__":
    launch(tyro.cli(AppConfig))
