import tyro

from egoexo_forge.api.robocap_hand_tracking import WilorEgoTrackingConfig, main

# Example usage
if __name__ == "__main__":
    main(
        tyro.cli(
            WilorEgoTrackingConfig,
            description="Run Wilor Ego Tracking.",
        )
    )
