import tyro

from gsplat_rust_renderer.apis.download import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
