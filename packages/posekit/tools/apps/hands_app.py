import gradio as gr
import tyro

from posekit.hands_app_ui.layout import build, launch
from posekit.hands_ui import Config, warm_up

config = tyro.cli(Config)
# `gradio` hot reload (posekit-hands-app-dev) re-runs this file on every edit: load the stacks once.
if gr.NO_RELOAD:
    warm_up()
demo = build()

if __name__ == "__main__":
    launch(demo, config)
