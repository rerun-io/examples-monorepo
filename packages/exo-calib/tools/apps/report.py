import tyro

from exo_calib.apis.report import main
from exo_calib.catalog_io import StageConfig

if __name__ == "__main__":
    main(tyro.cli(StageConfig, description="Stage E: score init and refined exo cameras against dataset ground truth."))
