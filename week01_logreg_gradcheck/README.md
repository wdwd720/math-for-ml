# Week 01 — Logistic Regression + Gradient Checking 

This week I built a full “calculus → code → verification → experiment” pipeline for **binary classification** using **logistic regression** (labels are `{-1, +1}`).

Steps:
- start from the logistic loss (margin view)
- derive gradients with the chain rule
- implement everything from scratch (NumPy only)
- **verify** gradients with finite-difference gradient checking
- train with gradient descent and visualize the results

This folder is a: derivation + code + tests + plots.

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
