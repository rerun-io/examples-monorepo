from wilor_nano.gradio_ui.hand_keypoint_ui import main

if __name__ == "__main__":
    demo = main()
    demo.launch(server_port=7861)
