"""DPVO visual odometry CLI demo."""

import tyro

from mini_dpvo.api.inference import DPVOInferenceConfig, DPVOPipelineHandle, run_dpvo_pipeline


def main(config: DPVOInferenceConfig) -> None:
    """Run the DPVO inference pipeline from the command line."""
    handle: DPVOPipelineHandle = DPVOPipelineHandle()

    for _msg in run_dpvo_pipeline(
        dpvo_config=config.dpvo_config,
        network_path=config.network_path,
        imagedir=config.imagedir,
        calib=config.calib,
        stride=config.stride,
        skip=config.skip,
        handle=handle,
    ):
        pass  # CLI exhausts the generator silently

    if handle.prediction is not None:
        print(f"Processed in {handle.elapsed_time:.2f}s")
        print(f"Keyframes: {handle.prediction.final_poses.shape[0]}")


if __name__ == "__main__":
    main(tyro.cli(DPVOInferenceConfig))
