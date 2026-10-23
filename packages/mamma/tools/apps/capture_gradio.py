from dagkit.form import gradio_app

from mamma.apis.capture import Config, build, shown_segment
from mamma.verify import example_cases

if __name__ == "__main__":
    gradio_app(Config, build, title="mamma capture", examples=example_cases(), segment=shown_segment).queue(max_size=2).launch()
