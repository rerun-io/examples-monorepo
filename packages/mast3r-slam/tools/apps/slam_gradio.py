from dagkit.app import gradio_app

from mast3r_slam.apis.slam import Config, build

if __name__ == "__main__":
    gradio_app(Config, build, title="MASt3R-SLAM").queue(max_size=2).launch()
