"""Thin CLI wrapper to launch the SAM3 chunk-based video demo."""

import tyro

from sam3_rerun.api.video_chunk import Sam3ChunkVideoDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3ChunkVideoDemoConfig))
