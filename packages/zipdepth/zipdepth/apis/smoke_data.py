"""Self-distillation smoke dataset: example frames pseudo-labelled by the released ZipDepth checkpoint.

Writes ``<out_dir>/rgb/<stem>.jpg`` and ``<out_dir>/depth/<stem>.png`` (full-range uint16, min-max
normalised relative inverse depth), then builds the memmap index ``LargeScaleDepthDataset`` needs.
The loss is scale-shift-invariant so absolute scale is irrelevant; uint16 PNG is used because the
``.npy`` depth path quantises ZipDepth's 0-0.15 output to ~37 levels.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from jaxtyping import Float32, UInt8
from monopriors.models.relative_depth.zipdepth import ZipDepthPredictor
from simplecv.video_io import VideoReader

from zipdepth.upstream_cli.prepare_index import build_index, convert_index

INDEX_ARTIFACTS = ("index.json", "index_rgb.npy", "index_depth.npy", "index_domain.npy", "index_metadata.json")


@dataclass
class SmokeDataConfig:
    out_dir: Path = Path("data/smoke")
    clip: Path = Path("assets/examples/clip.mp4")
    image: Path = Path("assets/examples/im0.jpg")
    num_frames: int = 32
    input_size: int = 768


def smoke_data_ready(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in INDEX_ARTIFACTS)


def build_smoke_data(config: SmokeDataConfig) -> None:
    rgb_dir = config.out_dir / "rgb"
    depth_dir = config.out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    teacher = ZipDepthPredictor(device="cuda", input_size=config.input_size)

    def write_pair(stem: str, bgr: UInt8[np.ndarray, "h w 3"]) -> None:
        rgb: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inv: Float32[np.ndarray, "h w"] = teacher(rgb, None).disparity
        inv_norm: Float32[np.ndarray, "h w"] = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)
        cv2.imwrite(str(rgb_dir / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(depth_dir / f"{stem}.png"), (inv_norm * 65535).astype(np.uint16))

    video = VideoReader(config.clip)
    for i, idx in enumerate(np.linspace(0, video.frame_cnt - 1, config.num_frames).astype(int)):
        write_pair(f"clip_{i:03d}", video.get_frame(int(idx)))
    im0 = cv2.imread(str(config.image))
    if im0 is None:
        raise FileNotFoundError(config.image)
    write_pair(config.image.stem, im0)

    build_index({"smoke": {"rgb": str(rgb_dir), "depth": str(depth_dir)}}, config.out_dir / "index.json")
    convert_index(config.out_dir / "index.json")
    if not smoke_data_ready(config.out_dir):
        raise RuntimeError(f"index conversion left {config.out_dir} incomplete")
    print(f"wrote {config.num_frames + 1} rgb/depth pairs + memmap index to {config.out_dir}")
