"""Shared model lifecycle for the multi-view Gradio applications."""

from monopriors.models.multiview.multiview_predictor import MultiviewPredictorCache

PREDICTOR_CACHE = MultiviewPredictorCache()
"""One atomic, single-resident predictor cache shared by both multi-view apps."""
