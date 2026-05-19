import tyro

from mv_api.api.full_exoego_app import main
from mv_api.api.full_exoego_pipeline import RRDPipelineConfig

if __name__ == "__main__":
    main(tyro.cli(RRDPipelineConfig))
