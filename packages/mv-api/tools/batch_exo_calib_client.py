"""CLI entry point for batch exo calibration.

Thin wrapper around mv_api.api.batch_calibration for CLI usage.
"""

import tyro

from mv_api.api.batch_calibration import BatchCalibConfig, run_batch_calibration

if __name__ == "__main__":
    run_batch_calibration(tyro.cli(BatchCalibConfig))
