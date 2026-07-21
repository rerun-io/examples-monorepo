"""Shared helpers for the multi-view Gradio applications."""

from pathlib import Path
from typing import Final

import gradio as gr

from monopriors.models.multiview.multiview_predictor import (
    ImagePreprocessingMode,
    MultiviewModelName,
    is_image_preprocessing_mode,
    is_multiview_model_name,
)

MULTIVIEW_EXAMPLE_SCENES: Final[tuple[tuple[str, str], ...]] = (
    ("car_landscape_12", "Car landscape"),
    ("bench", "Bench"),
    ("stairs", "Stairs"),
    ("tree", "Tree"),
)
IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpeg", ".jpg", ".png"})


def discover_multiview_examples(data_dir: Path) -> list[tuple[str, list[str]]]:
    """Return available example scenes in their stable display order."""
    examples: list[tuple[str, list[str]]] = []
    for directory_name, display_name in MULTIVIEW_EXAMPLE_SCENES:
        image_paths: list[str] = sorted(
            str(path)
            for path in (data_dir / directory_name).glob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if image_paths:
            view_label: str = "view" if len(image_paths) == 1 else "views"
            examples.append((f"{display_name} · {len(image_paths)} {view_label}", image_paths))
    return examples


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
