import tyro

from simplecv.apis.view_polycam_data import PolyViewConfig, view_polycam_data

if __name__ == "__main__":
    view_polycam_data(
        tyro.cli(
            PolyViewConfig,
            description="Visualize Polycam Data",
        )
    )
