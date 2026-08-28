"""End-to-end training smoke: data -> train -> resume -> torchrun -> checkpoints load in monopriors.

Every leg shares one argument list. Checkpoints go to a fresh directory that is renamed to
``<work_dir>/latest`` only after every leg passed, so a failed run never masquerades as a good one.
"""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from monopriors.models.relative_depth.zipdepth import ZipDepthPredictor
from monopriors.third_party.zipdepth.architecture import create_model
from monopriors.third_party.zipdepth.model_utils import strip_state_dict_prefixes

from zipdepth.apis.smoke_data import SmokeDataConfig, build_smoke_data, smoke_data_ready


@dataclass
class TrainSmokeConfig:
    data: SmokeDataConfig = field(default_factory=SmokeDataConfig)
    work_dir: Path = Path("data/smoke/runs")
    train_config: Path = Path("configs/default.json")
    height: int = 768
    width: int = 1024
    batch_size: int = 4
    num_workers: int = 2
    save_every_steps: int = 5


def _run(argv: list[str]) -> None:
    print("$", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def check_checkpoint(path: Path, rgb: np.ndarray) -> None:
    """A training checkpoint must load strictly into the vendored model and run through ZipDepthPredictor."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    missing_sections = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"} - set(ckpt)
    if missing_sections:
        raise RuntimeError(f"{path}: missing sections {sorted(missing_sections)}")
    create_model(variant="base").load_state_dict(strip_state_dict_prefixes(ckpt["model_state_dict"]), strict=True)
    pred = ZipDepthPredictor(device="cuda", checkpoint=path)(rgb, None)
    if pred.disparity.shape != rgb.shape[:2] or not np.isfinite(pred.disparity).all():
        raise RuntimeError(f"{path}: bad prediction {pred.disparity.shape}")
    print(f"OK {path}: strict key match, disparity range [{pred.disparity.min():.4f}, {pred.disparity.max():.4f}]")


def train_smoke(config: TrainSmokeConfig) -> None:
    if not smoke_data_ready(config.data.out_dir):
        build_smoke_data(config.data)

    config.work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=config.work_dir))
    single, ddp = run_dir / "single", run_dir / "ddp"
    common = [
        "-m", "zipdepth.upstream_cli.train",
        "--config", str(config.train_config),
        "--index-file", str(config.data.out_dir / "index.json"),
        "--height", str(config.height), "--width", str(config.width),
        "--batch-size", str(config.batch_size), "--num-workers", str(config.num_workers),
        "--save-every-steps", str(config.save_every_steps),
    ]  # fmt: skip
    _run([sys.executable, *common, "--epochs", "2", "--save-dir", str(single)])
    _run([sys.executable, *common, "--epochs", "2", "--save-dir", str(single), "--resume", str(single / "epoch_0.pth")])
    _run(["torchrun", "--nproc_per_node=1", *common, "--epochs", "1", "--save-dir", str(ddp)])

    bgr = cv2.imread(str(config.data.image))
    if bgr is None:
        raise FileNotFoundError(config.data.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    check_checkpoint(single / "final_model.pth", rgb)
    check_checkpoint(ddp / "final_model.pth", rgb)

    latest = config.work_dir / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    run_dir.rename(latest)
    print(f"smoke passed; checkpoints in {latest}")
