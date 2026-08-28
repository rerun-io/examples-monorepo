import tyro

from zipdepth.apis.infer_rerun import InferRerunConfig, infer_rerun

if __name__ == "__main__":
    infer_rerun(tyro.cli(InferRerunConfig))
