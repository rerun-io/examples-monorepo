import tyro

from vistadream.api.flux_outpainting import FluxOutpaintingConfig, main

if __name__ == "__main__":
    main(tyro.cli(FluxOutpaintingConfig, prog="flux_outpainting"))
