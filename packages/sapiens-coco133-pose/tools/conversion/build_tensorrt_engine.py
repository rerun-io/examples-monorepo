from __future__ import annotations

import tyro
from sapiens2_pose.api.coco133_tensorrt_conversion import build_tensorrt_engine


def main() -> None:
    """Run the TensorRT engine build CLI."""
    print(tyro.cli(build_tensorrt_engine))


if __name__ == "__main__":
    main()
