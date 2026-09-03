"""Tyro shim for static-batch ZipDepth-PromptDA TensorRT export."""

import tyro

from zipdepth.apis.prompted_trt import TrtExportConfig, main

if __name__ == "__main__":
    main(tyro.cli(TrtExportConfig))
