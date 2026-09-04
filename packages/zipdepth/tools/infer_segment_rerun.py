import tyro

from zipdepth.apis.infer_segment_rerun import InferSegmentRerunConfig, main

if __name__ == "__main__":
    main(tyro.cli(InferSegmentRerunConfig))
