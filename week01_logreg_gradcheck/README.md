# Week 01 — Logistic Regression + Gradient Checking (from scratch)

This week links AP Calculus chain rule intuition to modern ML optimization. Logistic regression composes linear scores with a nonlinear sigmoid, and the chain rule gives gradients used for learning. We start by building the project scaffold and a reproducible synthetic dataset, then use this as the base for implementing analytic and numerical gradients later in the week.

## What's inside
- `experiment.py` - smoke test that generates data and saves `data_preview.png`
- `derivation.md` - derivation scaffold for the Week 1 math
- `requirements.txt` - minimal dependencies (`numpy`, `matplotlib`)
- `src/utils.py` - data generation and stable sigmoid utilities
- `src/logreg.py` - logistic loss/gradient skeleton
- `src/gradcheck.py` - finite-difference gradient-check skeleton
- `tests/test_utils.py` - `unittest` checks for data and sigmoid behavior

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python experiment.py
```

## Expected output today
- Prints dataset shape and class counts
- Saves `data_preview.png`

## Next steps (later this week)
- Implement logistic regression loss and analytic gradients
- Implement finite-difference gradient checking
- Add training loop and diagnostics
