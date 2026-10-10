"""Require the production CUDA backend before collecting any project tests."""
import os

import pytest


def pytest_sessionstart(session):
    os.environ['NUMBERGRID_TEST_RUN'] = '1'
    requested = os.environ.get('JAX_PLATFORMS')
    if requested not in (None, 'cuda'):
        raise pytest.UsageError(
            'Project tests require JAX_PLATFORMS=cuda; CPU runs are not supported. '
            'Use bash scripts/run_gpu.sh -m pytest.')
    os.environ['JAX_PLATFORMS'] = 'cuda'

    import jax

    try:
        # Also reject an already-initialized non-GPU default backend.
        if jax.default_backend() != 'gpu' or not jax.devices('gpu'):
            raise RuntimeError('The default JAX backend is not a GPU')
    except RuntimeError as error:
        raise pytest.UsageError(
            'A working CUDA GPU is required for project tests; no CPU fallback. '
            'Use bash scripts/run_gpu.sh -m pytest.') from error
