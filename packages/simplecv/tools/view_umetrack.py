import tyro

from simplecv.apis.view_umetrack_data import UmeTrackVisualizeConfig, main

if __name__ == "__main__":
    main(
        tyro.cli(
            UmeTrackVisualizeConfig,
            description="Visualize UmeTrack datasets with hand pose annotations.",
        )
    )
