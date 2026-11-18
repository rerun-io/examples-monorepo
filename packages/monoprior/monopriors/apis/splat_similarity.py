"""Move a trained splat PLY by colsfm's ``prior_alignment`` similarity (``p_prior = s R p_colsfm + t``)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from serde import SerdeError, serde
from serde.json import from_json

from monopriors.splat_similarity import Similarity, transform_ply


@serde
@dataclass(frozen=True, slots=True)
class PriorAlignment:
    """colsfm ``summary.json`` ``prior_alignment``: the similarity from the colsfm frame to the pose-prior frame."""

    scale: float
    """Uniform scale s."""
    rotation_xyzw: list[float]
    """Rotation R as a unit quaternion (x, y, z, w)."""
    translation_m: list[float]
    """Translation t, metres of the prior frame."""

    def similarity(self) -> Similarity:
        """As a :class:`Similarity`."""
        return Similarity(
            scale=self.scale,
            rotation=Rotation.from_quat(self.rotation_xyzw).as_matrix(),
            translation=np.asarray(self.translation_m, dtype=np.float64),
        )


@serde
@dataclass(frozen=True, slots=True)
class ColsfmSummary:
    """The part of colsfm's ``summary.json`` read here (third-party to this tool: unknown fields allowed)."""

    prior_alignment: PriorAlignment
    """The alignment to the pose prior."""


def load_prior_alignment(path: Path) -> PriorAlignment:
    """Read ``prior_alignment`` from a colsfm ``summary.json``.

    Raises:
        ValueError: If the file is not valid JSON or has no well-formed ``prior_alignment``.
    """
    try:
        return from_json(ColsfmSummary, path.read_text()).prior_alignment
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: no readable prior_alignment ({error})") from error


@dataclass
class SplatSimilarityConfig:
    """Move an INRIA 3DGS PLY by the similarity in a colsfm ``summary.json``."""

    ply: Path
    """Input splat (trained in the unaligned colsfm frame)."""
    output_ply: Path
    """Output splat in the aligned frame."""
    colsfm_summary: Path
    """colsfm ``summary.json`` with ``prior_alignment``."""


def main(config: SplatSimilarityConfig) -> None:
    """Apply the alignment and write the moved PLY."""
    alignment: PriorAlignment = load_prior_alignment(config.colsfm_summary)
    count: int = transform_ply(config.ply, config.output_ply, alignment.similarity())
    config.output_ply.chmod(0o644)
    print(
        f"moved {count} splats by s={alignment.scale:.6f}, R=quat_xyzw{alignment.rotation_xyzw}, t={alignment.translation_m} -> {config.output_ply}"
    )
