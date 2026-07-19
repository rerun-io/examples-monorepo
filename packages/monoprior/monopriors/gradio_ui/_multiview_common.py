"""Shared parsing helpers for the multi-view Gradio applications."""

from typing import Literal

import gradio as gr

from monopriors.models.multiview.multiview_predictor import MultiviewModelName


def parse_multiview_model(model_name: str) -> MultiviewModelName:
    """Validate a Gradio string choice as a multi-view model backend."""
    if model_name == "vggt":
        return "vggt"
    if model_name == "g3t":
        return "g3t"
    raise gr.Error("Model must be vggt or g3t.")


def parse_preprocessing_mode(preprocessing_mode: str) -> Literal["crop", "pad"]:
    """Validate a Gradio string choice as a multi-view preprocessing mode."""
    if preprocessing_mode == "crop":
        return "crop"
    if preprocessing_mode == "pad":
        return "pad"
    raise gr.Error("Preprocessing mode must be crop or pad.")
