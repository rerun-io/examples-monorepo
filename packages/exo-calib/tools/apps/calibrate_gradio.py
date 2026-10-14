from dagkit.app import gradio_app

from exo_calib.apis.calibrate import Config, build
from exo_calib.verify import examples

if __name__ == "__main__":
    gradio_app(Config, build, title="exo-calib: calibrate the exo cameras of a catalog segment", examples=examples()).queue(max_size=2).launch()
