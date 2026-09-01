import tyro

from exo_calib.apis.report import ReportConfig, main

if __name__ == "__main__":
    main(tyro.cli(ReportConfig, description="Stage E: score init and refined exo cameras against dataset ground truth."))
