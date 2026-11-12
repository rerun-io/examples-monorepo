"""Empty stand-in for ``cholespy`` (dataforge-soma-x environments only).

py-soma-x 0.3.0 declares ``cholespy`` as a requirement, but no module of the ``soma`` package imports it: its
"cholespy" Laplacian solver is a pre-factored ``torch.linalg.cholesky``. cholespy ships no linux-aarch64 wheel and its
sdist does not build without a system BLAS/LAPACK, which blocked SOMA-X on the DGX Spark. The x86 wheel's own module
exposes no names either, so this package offers the same (empty) interface. Drop it when py-soma-x stops listing
cholespy or cholespy ships aarch64 wheels.
"""
