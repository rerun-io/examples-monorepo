if __name__ == "__main__":
    # Import under the guard: the backend and mp.Manager processes are spawned, and a spawned child re-imports
    # this file, so a top-level import would load gradio and the MASt3R checkpoint again in each of them per run.
    from mast3r_slam.gradio.mast3r_slam_ui import mast3r_slam_block as demo

    demo.queue(max_size=2).launch()
