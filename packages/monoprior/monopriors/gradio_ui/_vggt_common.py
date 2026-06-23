"""Shared helpers for the VGGT-based multiview Gradio UIs."""

from typing import Literal

import gradio as gr


def parse_preprocessing_mode(preprocessing_mode: str) -> Literal["crop", "pad"]:
    """Validate a Gradio string choice as a VGGT preprocessing mode."""

    if preprocessing_mode == "crop":
        return "crop"
    if preprocessing_mode == "pad":
        return "pad"
    raise gr.Error("Preprocessing mode must be crop or pad.")
