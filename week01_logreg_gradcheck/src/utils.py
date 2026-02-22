"""Utility functions for Week 01."""

from __future__ import annotations

import numpy as np


def make_blobs(n: int = 200, seed: int = 1):
    """Generate two Gaussian blobs for binary classification.

    Returns:
        X: Array of shape (n, 2)
        y: Array of shape (n,), labels in {-1, +1}
    """
    rng = np.random.default_rng(seed)

    mean_pos = np.array([2.0, 2.0])
    mean_neg = np.array([-2.0, -2.0])
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])

    n_pos = n // 2
    n_neg = n - n_pos

    x_pos = rng.multivariate_normal(mean_pos, cov, size=n_pos)
    x_neg = rng.multivariate_normal(mean_neg, cov, size=n_neg)

    X = np.vstack([x_pos, x_neg])
    y = np.concatenate([np.ones(n_pos, dtype=int), -np.ones(n_neg, dtype=int)])

    indices = rng.permutation(n)
    X = X[indices]
    y = y[indices]
    return X, y


def sigmoid(t):
    """Numerically stable sigmoid for NumPy arrays."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t, dtype=float)

    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))

    neg = ~pos
    exp_t = np.exp(t[neg])
    out[neg] = exp_t / (1.0 + exp_t)

    eps = np.finfo(float).eps
    return np.clip(out, eps, 1.0 - eps)
