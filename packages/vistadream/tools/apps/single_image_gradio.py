from dagkit.app import gradio_app

from vistadream.apis.single_image import Config, build

if __name__ == "__main__":
    gradio_app(Config, build, title="VistaDream single image").queue(max_size=2).launch()
