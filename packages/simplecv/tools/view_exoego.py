import tyro

from simplecv.apis.view_exoego import VisualizeConfig, main

# Example usage
if __name__ == "__main__":
    main(
        tyro.cli(
            VisualizeConfig,
            description="Visualize exo/ego datasets and relog static environment meshes when present.",
        )
    )
