import tyro

from simplecv.apis.view_torchcodec_multiview import TorchCodecMultiviewConfig, main

if __name__ == "__main__":
    main(tyro.cli(TorchCodecMultiviewConfig, description="Log videos and TorchCodec CUDA decoded frames to Rerun."))
