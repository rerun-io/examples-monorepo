"""DPVO visual odometry CLI demo."""

import tyro

from mini_dpvo.api.inference import DPVOInferenceConfig, dpvo_inference

if __name__ == "__main__":
    dpvo_inference(tyro.cli(DPVOInferenceConfig))
