"""Core logistic regression functions for Week 1."""

from __future__ import annotations

import numpy as np

from src.utils import sigmoid


def _validate_inputs(w, b, X, y):
    """Run shape checks and normalize input arrays."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (n, d).")
    if w.ndim != 1:
        raise ValueError("w must have shape (d,).")
    if y.ndim != 1:
        raise ValueError("y must have shape (n,).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows/samples.")
    if X.shape[1] != w.shape[0]:
        raise ValueError("w dimension must match the number of columns in X.")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("y must contain only labels in {-1, +1}.")

    b_arr = np.asarray(b, dtype=float)
    if b_arr.ndim > 0 and b_arr.size != 1:
        raise ValueError("b must be a scalar (or scalar-like array).")
    b_scalar = float(b_arr.reshape(-1)[0]) if b_arr.ndim > 0 else float(b_arr)
    return w, b_scalar, X, y


def loss_and_grads(w, b, X, y):
    """Return mean logistic loss and analytic gradients."""
    w, b, X, y = _validate_inputs(w=w, b=b, X=X, y=y)

    z = X @ w + b
    u = -y * z

    loss = float(np.mean(np.logaddexp(0.0, u)))
    s = sigmoid(u)
    factor = -y * s
    grad_w = (X.T @ factor) / X.shape[0]
    grad_b = float(np.mean(factor))

    return loss, grad_w, grad_b


def predict(w, b, X):
    """Predict -1/+1 labels from the score sign."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (n, d).")
    if w.ndim != 1:
        raise ValueError("w must have shape (d,).")
    if X.shape[1] != w.shape[0]:
        raise ValueError("w dimension must match the number of columns in X.")

    b_arr = np.asarray(b, dtype=float)
    if b_arr.ndim > 0 and b_arr.size != 1:
        raise ValueError("b must be a scalar (or scalar-like array).")
    b_scalar = float(b_arr.reshape(-1)[0]) if b_arr.ndim > 0 else float(b_arr)

    scores = X @ w + b_scalar
    return np.where(scores >= 0.0, 1.0, -1.0)
