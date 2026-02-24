"""Tests for Week 1 utility helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.utils import make_blobs, sigmoid


class TestUtils(unittest.TestCase):
    def test_make_blobs_shapes_and_labels(self):
        X, y = make_blobs(n=200, seed=1)

        self.assertEqual(X.shape, (200, 2))
        self.assertEqual(y.shape, (200,))

        labels = set(np.unique(y).tolist())
        self.assertEqual(labels, {-1, 1})

        self.assertEqual(int(np.sum(y == 1)), 100)
        self.assertEqual(int(np.sum(y == -1)), 100)

    def test_sigmoid_range(self):
        vals = sigmoid(np.array([-1000.0, 0.0, 1000.0]))

        for v in vals:
            self.assertGreater(v, 0.0)
            self.assertLess(v, 1.0)

        self.assertAlmostEqual(float(vals[1]), 0.5, places=12)
        self.assertLess(float(vals[0]), 1e-6)
        self.assertGreater(float(vals[2]), 1.0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
