import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    try:
        from beartype.claw import beartype_this_package

        beartype_this_package()
    except ImportError:
        pass

__all__ = [
    "api",
    "gradio_ui",
]
