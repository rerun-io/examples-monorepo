"""Open4DGS in the monorepo: 4D Gaussian people from Rerun catalog segments.

The TRiGS trainer (``trigs``) comes from ``recipes/open4dgs-trigs-src`` (upstream ASU-NEM-Labs/open4dgs-trigs at
ed7a84a). Its PyTorch3D calls (``knn_points``, ``sample_farthest_points``) import :mod:`open4dgs.pytorch3d_ops`
through the recipe's patch 0001, so PyTorch3D is never built. Catalog input, Rerun output, priors and held-out
evaluation live in this package.
"""
