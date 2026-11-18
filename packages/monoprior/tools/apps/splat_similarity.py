"""CLI shim: move a trained splat PLY by colsfm's prior_alignment similarity."""

import tyro

from monopriors.apis.splat_similarity import SplatSimilarityConfig, main

if __name__ == "__main__":
    main(tyro.cli(SplatSimilarityConfig))
