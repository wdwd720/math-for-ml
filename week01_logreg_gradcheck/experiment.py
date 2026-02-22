"""Week 1 smoke test: generate synthetic data and save a preview plot."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.utils import make_blobs


def main():
    X, y = make_blobs(n=200, seed=1)

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == -1))

    print("Week 1 setup OK.")
    print(f"X shape: {X.shape}")
    print(f'y counts: {{"+1": {pos_count}, "-1": {neg_count}}}')

    plt.figure()
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker="o")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker="x")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Synthetic 2D Classification Data")
    plt.savefig("data_preview.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plot: data_preview.png")


if __name__ == "__main__":
    main()
