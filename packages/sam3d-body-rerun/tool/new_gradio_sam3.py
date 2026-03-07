from sam3_rerun.gradio_ui.sam3_rerun_ui import TEST_INPUT_DIR, main

if __name__ == "__main__":
    demo = main()
    demo.launch(ssr_mode=False, allowed_paths=[str(TEST_INPUT_DIR)])
