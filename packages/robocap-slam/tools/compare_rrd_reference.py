"""Compare a Robocap candidate RRD against a saved reference RRD."""

import tyro

from robocap_slam.rrd_compare import RrdCompareConfig, RrdCompareReport, compare_rrd_reference


def main() -> None:
    """Run the RRD semantic comparison CLI."""
    config: RrdCompareConfig = tyro.cli(RrdCompareConfig)
    report: RrdCompareReport = compare_rrd_reference(config)
    print(
        f"RRD comparison passed: {report.rows} rows, "
        f"{len(report.columns)} compared columns, {len(report.skipped_columns)} skipped columns."
    )


if __name__ == "__main__":
    main()
