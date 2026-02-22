# Week 1 — Logistic Regression + Gradient Checking 

This week I wanted to understand logistic regression end-to-end, not just call a library. I generated a small synthetic 2D dataset (two Gaussian blobs labeled +1/−1), derived the gradients using the chain rule, implemented everything in NumPy, and then used finite-difference gradient checking to make sure my math matched my code.

My gradcheck max relative error was 3.587e−12 (so my gradients matched), and training dropped the loss from 0.693 → 0.0075 with 100% train/test accuracy on this simple synthetic dataset.

---

## What’s inside
- `derivation.md` - intuition + full gradient derivation (chain rule)
- `margin_plot.py` - plots logistic loss vs margin
- `experiment.py` - generates data, runs gradcheck, trains, evaluates, saves plots
- `src/utils.py` - dataset generation + numerically stable sigmoid
- `src/logreg.py` - logistic loss + analytic gradients + prediction
- `src/gradcheck.py` - central-difference gradient checker
- `tests/test_utils.py` - tests for utilities
- `tests/test_logreg.py` - tests for loss/gradients + gradcheck sanity

---

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
