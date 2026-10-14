from dagkit.app import gradio_app

from robocap_live.apis.hands_layer import Config, build, examples

if __name__ == "__main__":
    gradio_app(Config, build, title="RoboCap hands layer", examples=examples()).queue(max_size=2).launch()
