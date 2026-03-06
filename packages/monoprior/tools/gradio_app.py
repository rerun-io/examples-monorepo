import gradio as gr

from monopriors.gradio_ui.depth_compare_ui import relative_compare_block
from monopriors.gradio_ui.depth_inference_ui import depth_inference_block

title = "# Depth Comparison"
description1 = (
    """Demo to help compare different depth models. Including both Scale | Shift Invariant and Metric Depth types."""
)
description2 = """Invariant models mean they have no true scale and are only relative, where as Metric models have a true scale and are absolute (meters)."""
description3 = """Checkout the [Github Repo](https://github.com/pablovela5620/monoprior) [![GitHub Repo stars](https://img.shields.io/github/stars/pablovela5620/monoprior)](https://github.com/pablovela5620/monoprior)"""
model_load_status: str = "Models loaded and ready to use!"

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    gr.Markdown(description2)
    gr.Markdown(description3)
    gr.Markdown("### Depth Prediction demo")
    with gr.Tab(label="Depth Comparison"):
        relative_compare_block.render()
    with gr.Tab(label="Depth Inference"):
        depth_inference_block.render()

if __name__ == "__main__":
    demo.queue().launch()
