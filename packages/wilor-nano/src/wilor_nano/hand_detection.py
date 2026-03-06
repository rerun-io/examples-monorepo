"""Single-person hand detection utilities.

This module assumes the scene contains at most one person's hands per frame.
Concretely, it will detect at most one left hand and at most one right hand,
so outcomes are one of: both left and right present, only left present, only
right present, or none. It does not attempt multi-person disambiguation or
tracking, and therefore will not return multiple boxes for the same class.

Implementation notes:
- Uses an Ultralytics YOLO model and selects the top-1 box per class by
  confidence on-device.
- Arrays are annotated with jaxtyping for explicit dtype and shape.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results


@dataclass
class DetectionResult:
    left_xyxy: Float[ndarray, "1 4"] | None = None
    right_xyxy: Float[ndarray, "1 4"] | None = None
    wholebody_xyxy: Float[ndarray, "1 4"] | None = None


@dataclass
class HandDetectorConfig:
    verbose: bool = False
    hf_wilor_repo_id: str = "pablovela5620/wilor-nano"
    pretrained_dir: Path = Path.cwd() / "pretrained_models"


class HandDetector:
    def __init__(self, cfg: HandDetectorConfig) -> None:
        self.cfg: HandDetectorConfig = cfg
        self.init_models()

    def init_models(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg.pretrained_dir.mkdir(parents=True, exist_ok=True)
        yolo_model_path: Path = self.cfg.pretrained_dir / "detector.pt"

        # Download model if not exists
        if not yolo_model_path.exists():
            downloaded_path: str = hf_hub_download(
                repo_id=self.cfg.hf_wilor_repo_id,
                subfolder="pretrained_models",
                filename="detector.pt",
                local_dir=self.cfg.pretrained_dir,
            )
            yolo_model_path = Path(downloaded_path)
        self.hand_detector: YOLO = YOLO(yolo_model_path)
        self.hand_detector.to(self.device)

    @torch.no_grad()
    def __call__(self, rgb_hw3: UInt8[ndarray, "H W 3"], hand_conf: float) -> DetectionResult:
        """
        Detect hands in an RGB image using YOLO and select top-1 per class.

        - Uses vectorized access via Results.boxes to avoid per-box CPU transfers.
        - Returns at most one left and one right bbox (shape (1,4) each) if present.
        """
        # from ultralytics detector
        res: Results = self.hand_detector(
            rgb_hw3,
            conf=hand_conf,
            max_det=20,  # defensive cap; not used for class balancing
            classes=[0, 1],  # left/right; harmless if model already limited
            verbose=self.cfg.verbose,
        )[0]

        out: DetectionResult = DetectionResult()

        b: Boxes | None = res.boxes
        if b is None:
            return out

        # Vectorized tensors on model device
        # NOTE: Boxes.xyxy returns Tensor | ndarray; on GPU it's always Tensor
        xyxy: Float[Tensor, "n 4"] = torch.as_tensor(b.xyxy)
        conf_t: Tensor = torch.as_tensor(b.conf)  # (N,1) or (N,)
        cls_t_raw: Tensor = torch.as_tensor(b.cls)  # (N,1) or (N,)

        # Normalize shapes and dtypes
        conf: Float[Tensor, "n"] = conf_t.view(-1)
        cls: Int[Tensor, "n"] = cls_t_raw.view(-1).to(torch.int64)

        # Left hand (class 0)
        m_left: Tensor = (cls == 0) & (conf >= hand_conf)
        if torch.any(m_left):
            idx_rel_left: int = int(torch.argmax(conf[m_left]).item())
            left_xyxy_t: Float[Tensor, "4"] = xyxy[m_left][idx_rel_left]
            left_xyxy: Float[ndarray, "1 4"] = left_xyxy_t.unsqueeze(0).detach().cpu().numpy()
            out.left_xyxy = left_xyxy

        # Right hand (class 1)
        m_right: Tensor = (cls == 1) & (conf >= hand_conf)
        if torch.any(m_right):
            idx_rel_right: int = int(torch.argmax(conf[m_right]).item())
            right_xyxy_t: Float[Tensor, "4"] = xyxy[m_right][idx_rel_right]
            right_xyxy: Float[ndarray, "1 4"] = right_xyxy_t.unsqueeze(0).detach().cpu().numpy()
            out.right_xyxy = right_xyxy

        return out
