from __future__ import annotations

import tyro

from sapiens_coco133_pose.api.benchmark import run_pose_benchmark


def main() -> None:
    """Run the COCO-133 pose benchmark CLI."""
    print(tyro.cli(run_pose_benchmark))


if __name__ == "__main__":
    main()
