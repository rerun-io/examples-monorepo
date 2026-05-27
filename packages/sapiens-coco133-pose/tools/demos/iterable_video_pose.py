from __future__ import annotations

import tyro

from sapiens_coco133_pose.api.iterable import IterableVideoPoseConfig, run_iterable_video_pose_coco133


def main() -> None:
    """Run the iterable video pose demo."""
    print(run_iterable_video_pose_coco133(tyro.cli(IterableVideoPoseConfig)))


if __name__ == "__main__":
    main()
