import tyro

from simplecv.apis.view_torchcodec_singleview import TorchCodecSingleviewConfig, main

if __name__ == "__main__":
    main(tyro.cli(TorchCodecSingleviewConfig, description="Log video and TorchCodec CUDA decoded frames to Rerun."))
