"""CLI for writing Sapiens2 video pose results to a Rerun recording."""

import tyro

from sapiens2_pose.api.video import SapiensVideoPoseConfig, write_video_pose_rrd


def main() -> None:
    """Parse CLI config and run video pose inference."""
    config: SapiensVideoPoseConfig = tyro.cli(SapiensVideoPoseConfig)
    write_video_pose_rrd(config)


if __name__ == "__main__":
    main()
