import tyro

from exo_calib.apis.sweep import SweepConfig, main

if __name__ == "__main__":
    main(tyro.cli(SweepConfig, description="Hill-climb sweep over refine configurations, scored against GT extrinsics."))
