import tyro

from rerun_prompt_da.apis.prompt_da_trt_polycam import (
    PDATrtPolycamConfig,
    pda_trt_polycam_inference,
)

if __name__ == "__main__":
    pda_trt_polycam_inference(tyro.cli(PDATrtPolycamConfig))
