import tyro

from gsplat_rust_renderer.apis.train import Config, main

if __name__ == "__main__":
    config, extra_flags = tyro.cli(Config, return_unknown_args=True)
    main(config, extra_flags)
