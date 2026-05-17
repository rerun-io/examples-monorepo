"""CLI for writing Sapiens2 image pose results to Rerun and NumPy artifacts."""

import tyro

from sapiens2_pose.api.image_pose import ImagePoseConfig, run_image_pose


def main() -> None:
    """Parse CLI config and run single-image pose inference."""
    config: ImagePoseConfig = tyro.cli(ImagePoseConfig)
    run_image_pose(config)


if __name__ == "__main__":
    main()
