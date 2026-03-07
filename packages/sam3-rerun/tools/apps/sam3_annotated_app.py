"""Launch the SAM3 segmentation Gradio app with annotated image output."""

from sam3_rerun.gradio_ui.sam3_annotated_ui import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
