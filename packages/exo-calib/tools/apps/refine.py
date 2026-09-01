import tyro

from exo_calib.apis.refine import RefineConfig, main

if __name__ == "__main__":
    main(tyro.cli(RefineConfig, description="Stage C+D: correspondences, robust triangulation, and kornia-rs staged bundle adjustment."))
