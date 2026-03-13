"""Launch the SAM3 segmentation Gradio app with Rerun viewer."""

from sam3_rerun.gradio_ui.sam3_rerun_ui import TEST_INPUT_DIR, main

if __name__ == "__main__":
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True, allowed_paths=[str(TEST_INPUT_DIR)])
