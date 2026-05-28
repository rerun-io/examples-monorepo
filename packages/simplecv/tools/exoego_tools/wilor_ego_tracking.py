import tyro

from simplecv.apis.exoego_tools.wilor_ego_tracking import WilorEgoTrackingConfig, main

# Example usage
if __name__ == "__main__":
    main(
        tyro.cli(
            WilorEgoTrackingConfig,
            description="Run Wilor Ego Tracking.",
        )
    )
