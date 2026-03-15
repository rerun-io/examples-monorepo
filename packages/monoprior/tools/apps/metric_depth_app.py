from monopriors.gradio_ui.metric_depth_ui import main

demo = main()

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
