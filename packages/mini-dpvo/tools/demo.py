from argparse import ArgumentParser
import rerun as rr
from mini_dpvo.api.inference import inference_dpvo
from mini_dpvo.config import cfg as base_cfg


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--network-path", type=str, default="checkpoints/dpvo.pth")
    parser.add_argument("--imagedir", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--buffer", type=int, default=2048)
    parser.add_argument("--config", default="config/fast.yaml")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini_dpvo")

    base_cfg.merge_from_file(args.config)
    base_cfg.BUFFER_SIZE = args.buffer

    print("Running with config...")
    print(base_cfg)

    inference_dpvo(
        cfg=base_cfg,
        network_path=args.network_path,
        imagedir=args.imagedir,
        calib=args.calib,
        stride=args.stride,
        skip=args.skip,
    )
    rr.script_teardown(args)
