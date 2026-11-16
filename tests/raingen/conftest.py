"""Shared fixtures for the raingen test suite."""

import os

# One BLAS/FFT thread per test process: renders are many small array ops, where
# thread pools only add overhead (and oversubscribe CPUs under pytest-xdist).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pytest

SR = 48_000


@pytest.fixture
def sr():
    return SR


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def scene():
    """A fresh default Scene."""
    from raingen.scene import Scene
    return Scene()


@pytest.fixture
def tmp_out(tmp_path):
    return tmp_path
