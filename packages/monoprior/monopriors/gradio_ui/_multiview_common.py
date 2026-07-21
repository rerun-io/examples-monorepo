"""Shared parsing helpers for the multi-view Gradio applications."""

import gradio as gr

from monopriors.models.multiview.multiview_predictor import (
    ImagePreprocessingMode,
    MultiviewModelName,
    is_image_preprocessing_mode,
    is_multiview_model_name,
)


def parse_multiview_model(model_name: str) -> MultiviewModelName:
    """Validate a Gradio string choice as a multi-view model backend."""
    if is_multiview_model_name(model_name):
        return model_name
    raise gr.Error("Model must be vggt or g3t.")


def parse_preprocessing_mode(preprocessing_mode: str) -> ImagePreprocessingMode:
    """Validate a Gradio string choice as a multi-view preprocessing mode."""
    if is_image_preprocessing_mode(preprocessing_mode):
        return preprocessing_mode
    raise gr.Error("Preprocessing mode must be crop or pad.")
