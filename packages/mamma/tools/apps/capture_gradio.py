from dagkit.app import gradio_app

from mamma.apis.capture import Config, build
from mamma.verify import example_cases

if __name__ == "__main__":
    gradio_app(Config, build, title="mamma capture", examples=example_cases()).queue(max_size=2).launch()
