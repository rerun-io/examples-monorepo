from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from asmk import io_helpers
from jaxtyping import Float, Float32, Int, Int64
from mast3r.retrieval.model import how_select_local
from mast3r.retrieval.processor import Retriever
from numpy import ndarray
from torch import Tensor

if TYPE_CHECKING:
    from mast3r_slam.frame import Frame


class RetrievalDatabase(Retriever):
    """Image retrieval database backed by an ASMK inverted file.

    Wraps the MASt3R retrieval backbone with an online inverted file index
    so that new keyframes can be added and queried incrementally.
    """

    def __init__(
        self,
        modelname: str,
        backbone: object | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(modelname, backbone, device)

        self.ivf_builder = self.asmk.create_ivf_builder()

        self.kf_counter: int = 0
        self.kf_ids: list[int] = []

        self.query_dtype: torch.dtype = torch.float32
        self.query_device: str = device
        assert self.asmk.codebook is not None
        self.centroids: Float[Tensor, "n_centroids d"] = torch.from_numpy(self.asmk.codebook.centroids).to(
            device=self.query_device, dtype=self.query_dtype
        )

    # Mirrors forward_local in extract_local_features from retrieval/model.py
    def prep_features(
        self,
        backbone_feat: Float[Tensor, "1 n_patches feat_dim"],
    ) -> Float[Tensor, "1 n_local d"]:
        """Extract whitened local features from backbone tokens for retrieval.

        Args:
            backbone_feat: Raw MASt3R backbone feature tokens.

        Returns:
            Top-k whitened local features of shape (1, n_local, d).
        """
        retrieval_model = self.model

        # extract_features_and_attention without the encoding!
        backbone_feat_prewhitened = retrieval_model.prewhiten(backbone_feat)
        proj_feat = retrieval_model.projector(backbone_feat_prewhitened) + (
            0.0 if not retrieval_model.residual else backbone_feat_prewhitened
        )
        attention = retrieval_model.attention(proj_feat)
        proj_feat_whitened = retrieval_model.postwhiten(proj_feat)

        # how_select_local in
        topk_features: Float[Tensor, "1 n_local d"]
        topk_features, _, _ = how_select_local(
            proj_feat_whitened, attention, retrieval_model.nfeat
        )

        return topk_features

    def update(
        self,
        frame: "Frame",
        add_after_query: bool,
        k: int,
        min_thresh: float = 0.0,
    ) -> list[int]:
        """Query the database for similar keyframes and optionally add the frame.

        Args:
            frame: Frame whose features will be queried/added.
            add_after_query: If True, add the frame to the database after querying.
            k: Number of top-k results to return.
            min_thresh: Minimum score threshold for a valid retrieval.

        Returns:
            List of keyframe indices (in the database's ID space) that match.
        """
        assert frame.feat is not None
        feat: Float[Tensor, "1 n_local d"] = self.prep_features(frame.feat)
        id: int = self.kf_counter  # Using own counter since otherwise messes up IVF

        feat_np: Float32[ndarray, "n_local d"] = feat[0].cpu().numpy()  # Assumes one frame at a time!
        id_np: Int64[ndarray, "n_local"] = id * np.ones(feat_np.shape[0], dtype=np.int64)

        database_size: int = int(self.ivf_builder.ivf.n_images)
        # print("Database size: ", database_size, self.kf_counter)

        # Only query if already an image
        topk_image_inds: list[int] = []
        topk_codes: Int64[ndarray, "n_local k"] | None = None  # Change this if actualy querying
        if self.kf_counter > 0:
            ranks: Float[ndarray, "1 n_images"]
            ranked_scores: Float[ndarray, "1 n_images"]
            ranks, ranked_scores, topk_codes = self.query(feat_np, id_np)

            scores: Float[ndarray, "1 n_images"] = np.empty_like(ranked_scores)
            scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
            scores_tensor: Float[Tensor, "n_images"] = torch.from_numpy(scores)[0]

            topk_images = torch.topk(scores_tensor, min(k, database_size))

            valid: Float[Tensor, "k"] = topk_images.values > min_thresh
            topk_image_inds_tensor: Int[Tensor, "n_valid"] = topk_images.indices[valid]
            topk_image_inds = topk_image_inds_tensor.tolist()

        if add_after_query:
            self.add_to_database(feat_np, id_np, topk_codes)

        return topk_image_inds

    # The reason we need this function is becasue kernel and inverted file not defined when manually updating ivf_builder
    def query(
        self,
        feat: Float32[ndarray, "n_local d"],
        id: Int64[ndarray, "n_local"],
    ) -> tuple[Float[ndarray, "..."], Float[ndarray, "..."], Int64[ndarray, "..."]]:
        """Query the ASMK inverted file for matching images.

        Args:
            feat: 2-D array of local descriptors (n_local, d).
            id: 1-D array of image IDs for the query descriptors.

        Returns:
            A tuple of (ranks, scores, topk_codes) from the ASMK search.
        """
        step_params: dict = self.asmk.params.get("query_ivf")

        images2: Float[ndarray, "..."]
        ranks: Float[ndarray, "..."]
        scores: Float[ndarray, "..."]
        topk: Int64[ndarray, "..."]
        images2, ranks, scores, topk = self.accumulate_scores(
            self.asmk.codebook,
            self.ivf_builder.kernel,
            self.ivf_builder.ivf,
            feat,
            id,
            params=step_params,
        )

        return ranks, scores, topk

    def add_to_database(
        self,
        feat_np: Float32[ndarray, "n_local d"],
        id_np: Int64[ndarray, "n_local"],
        topk_codes: Int64[ndarray, "n_local k"] | None,
    ) -> None:
        """Add descriptors to the inverted file and update bookkeeping.

        Args:
            feat_np: 2-D array of local descriptors.
            id_np: 1-D array of image IDs.
            topk_codes: Pre-computed quantisation codes (reused from query), or None.
        """
        self.add_to_ivf_custom(feat_np, id_np, topk_codes)

        # Bookkeeping
        self.kf_ids.append(id_np[0])
        self.kf_counter += 1

    def quantize_custom(
        self,
        qvecs: Float[Tensor, "n_local d"],
        params: dict,
    ) -> Int[Tensor, "n_local k"]:
        """Quantise descriptors to the nearest centroids using L2 distance.

        Args:
            qvecs: Query descriptor vectors.
            params: ASMK quantisation parameters (contains ``multiple_assignment``).

        Returns:
            Top-k centroid indices per descriptor.
        """
        # Using trick for efficient distance matrix
        l2_dists: Float[Tensor, "n_local n_centroids"] = (
            torch.sum(qvecs**2, dim=1)[:, None]
            + torch.sum(self.centroids**2, dim=1)[None, :]
            - 2 * (qvecs @ self.centroids.mT)
        )
        k: int = params["quantize"]["multiple_assignment"]
        topk = torch.topk(l2_dists, k, dim=1, largest=False)
        return topk.indices

    def accumulate_scores(
        self,
        cdb: Any,
        kern: Any,
        ivf: Any,
        qvecs: Float32[ndarray, "n_local d"],
        qimids: Int64[ndarray, "n_local"],
        params: dict,
    ) -> tuple[Float[ndarray, "..."], Float[ndarray, "..."], Float[ndarray, "..."], Int64[ndarray, "..."]]:
        """Accumulate scores for every query image given codebook, kernel,
        inverted_file and parameters.

        Args:
            cdb: ASMK codebook.
            kern: ASMK kernel.
            ivf: ASMK inverted file.
            qvecs: 2-D array of local descriptors.
            qimids: 1-D array of query image IDs.
            params: ASMK query parameters.

        Returns:
            A tuple of (imids_all, ranks_all, scores_all, topk_all).
        """
        similarity_func = lambda *x: kern.similarity(*x, **params["similarity"])

        acc: list[tuple] = []
        slices: list = list(io_helpers.slice_unique(qimids))
        for imid, seq in slices:
            # Calculate qvecs to centroids distance matrix (without forming diff!)
            qvecs_torch: Float[Tensor, "n_local d"] = torch.from_numpy(qvecs[seq]).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds: Int[Tensor, "n_local k"] = self.quantize_custom(qvecs_torch, params)
            topk_inds_np: Int64[ndarray, "n_local k"] = topk_inds.cpu().numpy()
            quantized: tuple = (qvecs, topk_inds_np)

            aggregated = kern.aggregate_image(*quantized, **params["aggregate"])
            ranks: Float[ndarray, "..."]
            scores: Float[ndarray, "..."]
            ranks, scores = ivf.search(
                *aggregated, **params["search"], similarity_func=similarity_func
            )
            acc.append((imid, ranks, scores, topk_inds_np))

        imids_all: tuple
        ranks_all: tuple
        scores_all: tuple
        topk_all: tuple
        imids_all, ranks_all, scores_all, topk_all = zip(*acc)

        return (
            np.array(imids_all),
            np.vstack(ranks_all),
            np.vstack(scores_all),
            np.vstack(topk_all),
        )

    def add_to_ivf_custom(
        self,
        vecs: Float32[ndarray, "n_local d"],
        imids: Int64[ndarray, "n_local"],
        topk_codes: Int64[ndarray, "n_local k"] | None = None,
    ) -> None:
        """Add descriptors and corresponding image ids to the IVF.

        Args:
            vecs: 2-D array of local descriptors.
            imids: 1-D array of image ids.
            topk_codes: Pre-computed quantisation codes, or None to compute fresh.
        """
        ivf_builder = self.ivf_builder

        step_params: dict = self.asmk.params.get("build_ivf")

        if topk_codes is None:
            qvecs_torch: Float[Tensor, "n_local d"] = torch.from_numpy(vecs).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds: Int[Tensor, "n_local k"] = self.quantize_custom(qvecs_torch, step_params)
            topk_inds_np: Int64[ndarray, "n_local k"] = topk_inds.cpu().numpy()
        else:
            # Reuse previously calculated! Only take top 1
            # NOTE: Assuming build params multiple assignment is less than query
            k: int = step_params["quantize"]["multiple_assignment"]
            topk_inds_np = topk_codes[:, :k]

        quantized: tuple = (vecs, topk_inds_np, imids)

        aggregated = ivf_builder.kernel.aggregate(
            *quantized, **ivf_builder.step_params["aggregate"]
        )
        ivf_builder.ivf.add(*aggregated)
