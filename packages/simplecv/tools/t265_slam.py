import os
import sys

import tyro

# Ensure project root is on sys.path when running as a script
_THIS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from simplecv.apis.t265_slam import T265Config, main  # noqa: E402

# Example usage
if __name__ == "__main__":
    sys.exit(
        main(
            tyro.cli(
                T265Config,
                description="T265 SLAM Configuration",
            )
        )
    )
