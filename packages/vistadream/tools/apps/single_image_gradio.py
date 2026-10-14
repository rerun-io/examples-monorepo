from dagkit.app import gradio_app
from simplecv.rerun_log_utils import RerunTyroConfig

from vistadream.apis.single_image import Config, build, short_case

if __name__ == "__main__":
    gradio_app(Config, build, title="VistaDream single image", examples=[short_case(RerunTyroConfig(headless=True))]).queue(max_size=2).launch()
