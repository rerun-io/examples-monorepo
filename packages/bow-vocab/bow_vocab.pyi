"""Seeded hierarchical k-means vocabularies for float descriptors (Rust: kornia-staging's ``kornia_staging_bow::vocabulary``).

``Vocabulary`` trains a tree on a descriptor sample (k-means++ then Lloyd iterations per node; the same seed gives the
same tree for any thread count) and maps descriptors to words by greedy descent. ``Vocabulary.index`` turns a set of
images into TF-IDF bags of words (``ln(n_images / df)``, L1-normalized) in an inverted-file database, which scores
image pairs with DBoW2's L1 score ``sum_w min(q_w, d_w)``. Descriptor arrays must be C-contiguous float32 with 64, 128
or 256 columns; they are borrowed, not copied. Every refusal raises ``ValueError`` (``IndexError`` for an image index).
"""

from jaxtyping import Float32, UInt32
from numpy import ndarray

class Vocabulary:
    """A vocabulary tree trained once per run."""

    def __init__(
        self, sample: Float32[ndarray, "n_sample dim"], branching: int = 10, depth: int = 5, iterations: int = 10, seed: int = 0
    ) -> None:
        """Train on ``sample``; ``depth`` is lowered when ``branching ** depth`` exceeds the sample size."""
    @property
    def branching(self) -> int: ...
    @property
    def depth(self) -> int:
        """Levels actually built."""
    @property
    def n_words(self) -> int:
        """``branching ** depth``, including leaves no descriptor reached."""
    @property
    def descriptor_dim(self) -> int: ...
    def words(self, descriptors: Float32[ndarray, "n dim"]) -> UInt32[ndarray, " n"]:
        """The word of every descriptor."""
    def index(self, images: list[Float32[ndarray, "_ dim"]]) -> BowIndex:
        """Bags of words of ``images`` (entry ``i`` is ``images[i]``) in an inverted-file database."""

class BowIndex:
    """TF-IDF bags of words of a fixed set of images."""

    def __len__(self) -> int: ...
    def similarity(self) -> Float32[ndarray, "n n"]:
        """Every pair's L1 score, symmetric, 1.0 on the diagonal, in ``[0, 1]``."""
    def score(self, a: int, b: int) -> float:
        """The L1 score of images ``a`` and ``b``."""
    def query(self, i: int, top_k: int) -> list[tuple[int, float]]:
        """The ``top_k`` images most similar to image ``i`` (itself excluded), best first."""
