"""Plot logistic loss as a function of margin."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def main():
    m = np.linspace(-8.0, 8.0, 400)
    loss = np.logaddexp(0.0, -m)

    plt.figure()
    plt.plot(m, loss)
    plt.axvline(0.0)
    plt.xlabel("margin m")
    plt.ylabel("loss")
    plt.title("Logistic Loss vs Margin")
    plt.savefig("loss_vs_margin.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: loss_vs_margin.png")


if __name__ == "__main__":
    main()
