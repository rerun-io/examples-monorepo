from mini_dpvo.gradio_ui.dpvo_ui import dpvo_block

demo = dpvo_block

if __name__ == "__main__":
    demo.queue(max_size=2).launch(ssr_mode=False)
