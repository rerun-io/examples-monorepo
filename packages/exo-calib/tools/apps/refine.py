import tyro

from exo_calib.apis.refine import main
from exo_calib.catalog_io import StageConfig

if __name__ == "__main__":
    main(tyro.cli(StageConfig, description="Stage C+D: correspondences, robust triangulation, and kornia-rs staged bundle adjustment."))
