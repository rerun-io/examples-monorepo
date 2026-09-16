"""Import a known Basalt sensor calibration into the catalog."""

import tyro

from slam_rs.apis.import_imu_calibration import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
