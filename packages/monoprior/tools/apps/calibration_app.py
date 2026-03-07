import gradio as gr

from monopriors.gradio_ui.multiview_calibration_ui import mv_calibration_block

with gr.Blocks() as demo:
    mv_calibration_block.render()

if __name__ == "__main__":
    demo.queue().launch()
