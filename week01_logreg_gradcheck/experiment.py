"""Week 1 experiment: gradcheck, training, and plots for logistic regression."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.gradcheck import gradcheck
from src.logreg import loss_and_grads, predict
from src.utils import make_blobs


def train_logreg(X_train, y_train, lr=0.1, steps=1000):
    """Run gradient descent and return learned parameters plus loss history."""
    n_features = X_train.shape[1]
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    loss_history = []

    for step in range(steps):
        loss, grad_w, grad_b = loss_and_grads(w, b, X_train, y_train)
        loss_history.append(loss)

        if step % 100 == 0:
            print(f"step {step} loss {loss:.6f}")

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, np.asarray(loss_history, dtype=float)


def save_loss_curve(loss_history, filename="loss_curve.png"):
    """Save loss-vs-step plot."""
    plt.figure()
    plt.plot(np.arange(loss_history.size), loss_history)
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.title("Training Loss Curve")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_decision_boundary(X, y, w, b, filename="decision_boundary.png"):
    """Save scatter plot with learned linear decision boundary."""
    plt.figure()
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker="o")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker="x")

    x_min = float(np.min(X[:, 0]) - 1.0)
    x_max = float(np.max(X[:, 0]) + 1.0)

    if abs(w[1]) > 1e-12:
        xs = np.array([x_min, x_max], dtype=float)
        ys = -(w[0] * xs + b) / w[1]
        plt.plot(xs, ys)
    elif abs(w[0]) > 1e-12:
        x_boundary = -b / w[0]
        plt.axvline(x_boundary)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    X, y = make_blobs(n=200, seed=1)

    split = int(0.8 * X.shape[0])
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    w0 = np.zeros(X_train.shape[1], dtype=float)
    b0 = 0.0

    def wrapped_loss_fn(params):
        w = np.asarray(params["w"], dtype=float)
        b = float(np.asarray(params["b"], dtype=float))
        loss, grad_w, grad_b = loss_and_grads(w, b, X_train, y_train)
        return loss, {"w": grad_w, "b": np.asarray(grad_b, dtype=float)}

    gc_result = gradcheck(
        loss_fn=wrapped_loss_fn,
        params={"w": w0.copy(), "b": np.asarray(b0, dtype=float)},
        eps=1e-5,
    )
    max_rel_error = gc_result["max_rel_error"]
    print(f"Gradcheck max rel error: {max_rel_error:.3e}")

    w, b, loss_history = train_logreg(X_train, y_train, lr=0.1, steps=1000)
    final_loss, _, _ = loss_and_grads(w, b, X_train, y_train)

    train_pred = predict(w, b, X_train)
    test_pred = predict(w, b, X_test)
    train_acc = float(np.mean(train_pred == y_train))
    test_acc = float(np.mean(test_pred == y_test))

    save_loss_curve(loss_history, filename="loss_curve.png")
    save_decision_boundary(X, y, w, b, filename="decision_boundary.png")

    print(f"Start loss: {loss_history[0]:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Saved plots: loss_curve.png, decision_boundary.png")


if __name__ == "__main__":
    main()
