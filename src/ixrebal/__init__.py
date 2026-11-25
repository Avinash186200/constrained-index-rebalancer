"""Constrained minimum-variance index rebalancing with an independent rule checker."""
import os as _os

# One BLAS thread: results must not depend on the machine's core count.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__version__ = "1.0.0"
