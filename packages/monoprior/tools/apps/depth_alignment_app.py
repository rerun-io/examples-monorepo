from monopriors.gradio_ui.depth_alignment_ui import main

demo = main()

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
