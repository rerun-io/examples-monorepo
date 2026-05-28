import tyro

from simplecv.apis.view_easymocap_data import ViewEasyMocapConfig, view_easymocap

# Example usage
if __name__ == "__main__":
    view_easymocap(
        tyro.cli(
            ViewEasyMocapConfig,
            description="Visualize EasyMocap Triangulation",
        )
    )
