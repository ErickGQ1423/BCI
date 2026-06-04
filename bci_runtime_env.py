"""Process-level defaults for running this BCI project."""

import os


NUMBA_CACHE_DIR = os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(NUMBA_CACHE_DIR, exist_ok=True)

MPLCONFIGDIR = os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
