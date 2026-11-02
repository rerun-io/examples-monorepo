import tyro

from gsplat_rust_renderer.apis.hexels_compare import CompareConfig, main

if __name__ == "__main__":
    main(tyro.cli(CompareConfig))
