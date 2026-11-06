from metrictrail.gradio_ui.metrictrail_ui import build_app

if __name__ == "__main__":
    build_app().queue(max_size=2).launch(ssr_mode=False)
