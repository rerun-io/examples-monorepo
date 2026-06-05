import tyro

from simplecv.apis.view_hocap_torchcodec_multiview import HocapTorchCodecViewConfig, main

if __name__ == "__main__":
    main(tyro.cli(HocapTorchCodecViewConfig, description="Log HOCAP videos and TorchCodec CUDA decoded frames to Rerun."))
