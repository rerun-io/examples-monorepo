"""Seeded hierarchical k-means vocabularies for float descriptors (Rust: kornia-staging's ``kornia_staging_bow::vocabulary``).

``Vocabulary`` trains a tree on a descriptor sample (k-means++ then Lloyd iterations per node; the same seed gives the
same tree for any thread count) and maps descriptors to words by greedy descent. ``Vocabulary.similarity`` scores
every pair of a set of images with DBoW2's L1 score ``sum_w min(q_w, d_w)`` over their TF-IDF bags of words
(``ln(n_images / df)``, L1-normalized). Descriptor arrays must be C-contiguous float32 with 64, 128 or 256 columns;
they are borrowed, not copied, and read with the GIL released, so do not write to them during a call. A wrong width
or layout (Fortran-ordered, strided) raises ``ValueError``; another dtype (float64) raises ``TypeError``.
"""

from jaxtyping import Float32, UInt32
from numpy import ndarray

class Vocabulary:
    """A vocabulary tree trained once per run."""

    def __init__(self, sample: Float32[ndarray, "n_sample dim"], *, branching: int, depth: int, iterations: int, seed: int) -> None:
        """Train on ``sample``; ``depth`` is lowered when ``branching ** depth`` exceeds the sample size."""
    @property
    def n_words(self) -> int:
        """``branching ** depth`` (depth as built), including leaves no descriptor reached."""
    def words(self, descriptors: Float32[ndarray, "n dim"]) -> UInt32[ndarray, " n"]:
        """The word of every descriptor."""
    def similarity(self, images: list[Float32[ndarray, "_ dim"]]) -> Float32[ndarray, "n n"]:
        """Every pair's L1 score, symmetric, 1.0 on the diagonal, 0 for an image with an empty bag, in ``[0, 1]``."""
