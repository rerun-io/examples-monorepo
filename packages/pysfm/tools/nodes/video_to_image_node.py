from pysfm.gradio_ui.nodes.video_to_image_ui import main

demo = main()

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
