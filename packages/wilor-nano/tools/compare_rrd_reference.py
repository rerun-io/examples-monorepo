import tyro

from wilor_nano.rrd_compare import RrdCompareConfig, RrdCompareReport, compare_rrd_reference


def main() -> None:
    config: RrdCompareConfig = tyro.cli(RrdCompareConfig)
    report: RrdCompareReport = compare_rrd_reference(config)
    print(
        f"RRD comparison passed: {report.rows} rows, "
        f"{len(report.columns)} compared columns, {len(report.skipped_columns)} skipped columns."
    )


if __name__ == "__main__":
    main()
