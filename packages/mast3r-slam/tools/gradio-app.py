from mast3r_slam.gradio.mast3r_slam_ui import mast3r_slam_block as demo

if __name__ == "__main__":
    demo.queue(max_size=2).launch()
