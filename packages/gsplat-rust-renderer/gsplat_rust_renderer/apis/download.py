"""Download one benchmark scene or the complete Blender corpus."""
from dataclasses import dataclass
from typing import Annotated, Literal

import tyro

from gsplat_rust_renderer.nerfbaselines import BLENDER_SCENES, BlenderScene, download_and_extract, download_tandt


@dataclass(frozen=True, slots=True)
class Config:
    kind: Annotated[Literal["data", "pretrained", "all", "tandt"], tyro.conf.Positional] = "pretrained"
    """Asset kind; all downloads datasets and checkpoints."""
    scene: Annotated[Literal[BlenderScene, "all"], tyro.conf.Positional] = "lego"
    """One Blender scene or all eight scenes; ignored for tandt."""


def main(config: Config) -> None:
    """Resolve the requested corpus and reuse idempotent per-scene downloads."""
    if config.kind == "tandt":
        download_tandt()
        return
    scenes: tuple[str, ...] = BLENDER_SCENES if config.scene == "all" else (config.scene,)
    kinds: tuple[str, ...] = ("data", "pretrained") if config.kind == "all" else (config.kind,)
    for kind in kinds:
        for scene in scenes:
            download_and_extract(kind, scene)
