"""Utilities for numerical gradient checking."""

from __future__ import annotations

import numpy as np


def _clone_params(params):
    """Create float copies of parameter values."""
    cloned = {}
    for key, value in params.items():
        arr = np.asarray(value, dtype=float)
        cloned[key] = np.array(arr, dtype=float, copy=True)
    return cloned


def _parse_loss_output(output):
    """Handle either loss-only or (loss, grads) output."""
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError("loss_fn tuple output must be (loss, grads_dict).")
        loss, grads = output
        if not isinstance(grads, dict):
            raise ValueError("loss_fn second return value must be a grads dict.")
        return float(loss), grads
    return float(output), None


def gradcheck(loss_fn, params, eps=1e-5):
    """Compare analytic gradients to central-difference gradients.

    loss_fn(params) can return either:
    1) loss (float), or
    2) (loss, grads_dict)
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not isinstance(params, dict) or len(params) == 0:
        raise ValueError("params must be a non-empty dict.")

    base_params = _clone_params(params)
    base_loss, analytic_raw = _parse_loss_output(loss_fn(_clone_params(base_params)))
    _ = base_loss

    numeric = {}
    for key, param_arr in base_params.items():
        num_grad = np.zeros_like(param_arr, dtype=float)
        for idx in np.ndindex(param_arr.shape):
            plus_params = _clone_params(base_params)
            minus_params = _clone_params(base_params)

            plus_params[key][idx] += eps
            minus_params[key][idx] -= eps

            loss_plus, _ = _parse_loss_output(loss_fn(plus_params))
            loss_minus, _ = _parse_loss_output(loss_fn(minus_params))
            num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        numeric[key] = num_grad

    analytic = {}
    rel_error = {}
    max_rel_error = 0.0

    if analytic_raw is not None:
        for key, param_arr in base_params.items():
            if key not in analytic_raw:
                raise ValueError(f"Analytic grads missing key '{key}'.")

            a = np.asarray(analytic_raw[key], dtype=float)
            if a.shape != param_arr.shape:
                raise ValueError(
                    f"Analytic grad for '{key}' has shape {a.shape}, "
                    f"expected {param_arr.shape}."
                )

            n = numeric[key]
            denom = np.maximum(1.0, np.maximum(np.abs(a), np.abs(n)))
            rel = np.abs(a - n) / denom

            analytic[key] = np.array(a, dtype=float, copy=True)
            rel_error[key] = rel
            max_rel_error = max(max_rel_error, float(np.max(rel)))
    else:
        max_rel_error = float("nan")

    return {
        "analytic": analytic,
        "numeric": numeric,
        "rel_error": rel_error,
        "max_rel_error": float(max_rel_error),
    }
