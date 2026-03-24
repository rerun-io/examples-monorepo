import time

import gradio as gr
import numpy as np
from jaxtyping import UInt8
from PIL import Image

from vistadream.ops.flux import FluxInpainting, FluxInpaintingConfig
from vistadream.resize_utils import add_border_and_mask, process_image


def build_ui(offload: bool = True) -> None:
    flux_inpainter: FluxInpainting = FluxInpainting(FluxInpaintingConfig(offload=offload))

    def get_res(image: Image.Image, expansion_percent: float) -> Image.Image:
        max_dimension: int = 1024
        image = process_image(image, max_dimension=max_dimension)

        width: int
        height: int
        width, height = image.size
        print(f"Resized image size: {width}x{height}")

        # Auto-generate outpainting setup: user-controlled border expansion
        border_percent: float = expansion_percent / 200.0
        print(f"Border expansion: {expansion_percent}% total ({border_percent * 100:.1f}% per side)")

        bordered_image: Image.Image
        mask: Image.Image
        bordered_image, mask = add_border_and_mask(
            image,
            zoom_left=border_percent,
            zoom_right=border_percent,
            zoom_up=border_percent,
            zoom_down=border_percent,
        )

        rgb_hw3: UInt8[np.ndarray, "h w 3"] = np.array(bordered_image)
        mask_hw: UInt8[np.ndarray, "h w"] = np.array(mask)

        t0: float = time.perf_counter()
        result: Image.Image = flux_inpainter(rgb_hw3=rgb_hw3, mask=mask_hw)
        t1: float = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")

        return result

    with gr.Blocks() as demo, gr.Column():
        input_image = gr.Image(
            type="pil",
            sources=["upload", "clipboard"],
            label="Upload Image for Outpainting",
        )

        expansion_slider = gr.Slider(
            minimum=5,
            maximum=50,
            value=20,
            step=1,
            label="Border Expansion (%)",
            info="Total percentage to expand the image (distributed evenly on all sides)",
        )

        run_btn = gr.Button("Run Outpainting")

        with gr.Row():
            flux_res = gr.Image(label="Outpainted Result")

        run_btn.click(fn=get_res, inputs=[input_image, expansion_slider], outputs=[flux_res])

    demo.launch()


if __name__ == "__main__":
    build_ui()
