import tyro

from exo_calib.apis.html_report import HtmlReportConfig, main

if __name__ == "__main__":
    main(tyro.cli(HtmlReportConfig, description="Render the init-vs-refined evaluation as a single-file HTML report."))
