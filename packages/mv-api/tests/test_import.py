"""Smoke tests for the mv_api package surface."""


def test_import_mv_api() -> None:
    import mv_api  # noqa: F401


def test_import_retained_modules() -> None:
    import mv_api.api.batch_calibration  # noqa: F401
    import mv_api.api.exo_only_calibration  # noqa: F401
    import mv_api.api.full_exoego_pipeline  # noqa: F401
    import mv_api.gradio_ui.full_pipeline_rrd_ui  # noqa: F401
