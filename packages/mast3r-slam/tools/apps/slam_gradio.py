from dagkit.form import gradio_app

from mast3r_slam.apis.slam import Config, build, shown_segment
from mast3r_slam.verify import examples

if __name__ == "__main__":
    gradio_app(Config, build, title="MASt3R-SLAM", examples=examples(), segment=shown_segment).queue(max_size=2).launch()
