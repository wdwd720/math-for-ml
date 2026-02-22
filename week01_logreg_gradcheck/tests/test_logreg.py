"""Unit tests for logistic regression and gradient checking."""

from __future__ import annotations

import unittest

import numpy as np

from src.gradcheck import gradcheck
from src.logreg import loss_and_grads
from src.utils import make_blobs


class TestLogReg(unittest.TestCase):
    def test_loss_and_grads_shapes(self):
        X = np.array(
            [[1.0, 2.0], [0.5, -1.0], [2.0, 0.2], [-1.0, -1.5], [0.1, 0.3]],
            dtype=float,
        )
        y = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=float)
        w = np.array([0.2, -0.1], dtype=float)
        b = 0.05

        loss, grad_w, grad_b = loss_and_grads(w, b, X, y)

        self.assertTrue(np.isfinite(loss))
        self.assertEqual(grad_w.shape, (2,))
        self.assertIsInstance(grad_b, float)

    def test_gradcheck_sanity(self):
        X, y = make_blobs(n=10, seed=7)
        rng = np.random.default_rng(0)
        w = rng.normal(loc=0.0, scale=0.1, size=X.shape[1])
        b = float(rng.normal(loc=0.0, scale=0.1))

        def wrapped_loss_fn(params):
            w_p = np.asarray(params["w"], dtype=float)
            b_p = float(np.asarray(params["b"], dtype=float))
            loss, grad_w, grad_b = loss_and_grads(w_p, b_p, X, y)
            return loss, {"w": grad_w, "b": np.asarray(grad_b, dtype=float)}

        result = gradcheck(
            loss_fn=wrapped_loss_fn,
            params={"w": w, "b": np.asarray(b, dtype=float)},
            eps=1e-5,
        )
        self.assertLess(result["max_rel_error"], 1e-3)


if __name__ == "__main__":
    unittest.main()
