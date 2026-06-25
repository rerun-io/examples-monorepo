import tyro

from gsplat_rust_renderer.apis.log_splats_with_cameras import LogSplatsWithCamerasConfig, main

if __name__ == "__main__":
    main(tyro.cli(LogSplatsWithCamerasConfig))
