import tyro

from mv_api.api.wilor_ego_tracking import WilorEgoTrackingConfig, main

# Example usage
if __name__ == "__main__":
    main(
        tyro.cli(
            WilorEgoTrackingConfig,
            description="Run Wilor Ego Tracking.",
        )
    )
