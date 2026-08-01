"""Lazy cache for calibration models unrelated to the multi-view backend."""

from _thread import LockType
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from sam3.api.predictor import SAM3Config, SAM3Predictor

from monopriors.models.relative_depth import get_relative_predictor
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor


@dataclass(frozen=True, slots=True)
class CalibrationAuxiliaryModels:
    """Optional SAM3 and MoGe dependencies requested by one run."""

    seg_predictor: SAM3Predictor | None
    moge_predictor: BaseRelativePredictor | None


class CalibrationAuxiliaryCache:
    """Load SAM3 and MoGe independently from multiview backend replacement."""

    def __init__(self) -> None:
        self._lock: LockType = Lock()
        self._segmenters: dict[str, SAM3Predictor] = {}
        self._depth_predictors: dict[str, BaseRelativePredictor] = {}

    def get(
        self,
        *,
        device: Literal["cuda", "cpu"],
        segment_people: bool,
        refine_depth_maps: bool,
    ) -> CalibrationAuxiliaryModels:
        """Return only the requested dependencies, constructing each once per device."""
        with self._lock:
            seg_predictor: SAM3Predictor | None = None
            if segment_people:
                if device not in self._segmenters:
                    self._segmenters[device] = SAM3Predictor(SAM3Config(device=device))
                seg_predictor = self._segmenters[device]

            moge_predictor: BaseRelativePredictor | None = None
            if refine_depth_maps:
                if device not in self._depth_predictors:
                    self._depth_predictors[device] = get_relative_predictor("MoGeV1Predictor")(device=device)
                moge_predictor = self._depth_predictors[device]

            return CalibrationAuxiliaryModels(
                seg_predictor=seg_predictor,
                moge_predictor=moge_predictor,
            )


AUXILIARY_MODEL_CACHE = CalibrationAuxiliaryCache()
