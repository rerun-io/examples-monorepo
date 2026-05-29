import tyro

from simplecv.apis.quest3_oakd_loader import Quest3VisualizeConfig, main

# Example usage
if __name__ == "__main__":
    main(
        tyro.cli(
            Quest3VisualizeConfig,
            description="Visualize Quest3 OakD datasets.",
        )
    )
