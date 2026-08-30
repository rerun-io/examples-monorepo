import tyro

from monopriors.gradio_ui.stereo_depth_ui import AppConfig, launch

if __name__ == "__main__":
    launch(tyro.cli(AppConfig))
