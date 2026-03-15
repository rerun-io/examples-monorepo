"""
Demonstrates integrating Rerun visualization with Gradio.

Provides example implementations of data streaming, keypoint annotation, and dynamic
visualization across multiple Gradio tabs using Rerun's recording and visualization capabilities.
"""

import os
import tempfile
import time
from collections import namedtuple
from math import cos, sin

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun

ColorGrid = namedtuple("ColorGrid", ["positions", "colors"])


def build_color_grid(x_count: int = 10, y_count: int = 10, z_count: int = 10, twist: int = 0) -> ColorGrid:
    """
    Create a cube of points with colors.

    The total point cloud will have x_count * y_count * z_count points.

    Parameters
    ----------
    x_count, y_count, z_count:
        Number of points in each dimension.
    twist:
        Angle to twist from bottom to top of the cube

    """
    grid = np.mgrid[
        slice(-x_count, x_count, x_count * 1j),
        slice(-y_count, y_count, y_count * 1j),
        slice(-z_count, z_count, z_count * 1j),
    ]

    angle = np.linspace(-float(twist) / 2, float(twist) / 2, z_count)
    for z in range(z_count):
        xv, yv, zv = grid[:, :, :, z]
        rot_xv = xv * cos(angle[z]) - yv * sin(angle[z])
        rot_yv = xv * sin(angle[z]) + yv * cos(angle[z])
        grid[:, :, :, z] = [rot_xv, rot_yv, zv]

    positions = np.vstack([xyz.ravel() for xyz in grid])

    colors = np.vstack(
        [
            xyz.ravel()
            for xyz in np.mgrid[
                slice(0, 255, x_count * 1j),
                slice(0, 255, y_count * 1j),
                slice(0, 255, z_count * 1j),
            ]
        ]
    )

    return ColorGrid(positions.T, colors.T.astype(np.uint8))


# However, if you have a workflow that creates an RRD file instead, you can still send it
# directly to the viewer by simply returning the path to the RRD file.
#
# This may be helpful if you need to execute a helper tool written in C++ or Rust that can't
# be easily modified to stream data directly via Gradio.
#
# In this case you may want to clean up the RRD file after it's sent to the viewer so that you
# don't accumulate too many temporary files.
@rr.thread_local_stream("rerun_example_cube_rrd")
def create_cube_rrd(x, y, z, pending_cleanup) -> str:
    cube = build_color_grid(int(x), int(y), int(z), twist=0)
    rr.log("cube", rr.Points3D(cube.positions, colors=cube.colors, radii=0.5))

    # Simulate delay
    time.sleep(x / 10)

    # We eventually want to clean up the RRD file after it's sent to the viewer, so tracking
    # any pending files to be cleaned up when the state is deleted.
    with tempfile.NamedTemporaryFile(prefix="cube_", suffix=".rrd", delete=False) as temp:
        pending_cleanup.append(temp.name)

        blueprint = rrb.Spatial3DView(origin="cube")
        rr.save(temp.name, default_blueprint=blueprint)

    # Just return the name of the file -- Gradio will convert it to a FileData object
    # and send it to the viewer.
    return temp.name


def cleanup_cube_rrds(pending_cleanup: list[str]) -> None:
    for f in pending_cleanup:
        os.unlink(f)


def render_vistadream_block() -> None:
    """Render the VistaDream Rerun section into the current Blocks context.

    Note: Does not create a new gr.Blocks() or additional Tabs. Intended to be
    called from within an existing layout (e.g., a Tab in the host app).
    """
    pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleanup_cube_rrds)
    with gr.Row():
        x_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="X Count")
        y_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="Y Count")
        z_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="Z Count")
    with gr.Row():
        create_rrd = gr.Button("Create RRD")
    with gr.Row():
        viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "hidden",
                "selection": "hidden",
            },
        )
    create_rrd.click(
        create_cube_rrd,
        inputs=[x_count, y_count, z_count, pending_cleanup],
        outputs=[viewer],
    )
