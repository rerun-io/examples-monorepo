from dagkit.form import gradio_app

from exo_calib.apis.calibrate import Config, build, shown_segment
from exo_calib.verify import examples

if __name__ == "__main__":
    gradio_app(Config, build, title="exo-calib: calibrate the exo cameras of a catalog segment", examples=examples(), segment=shown_segment).queue(max_size=2).launch()
