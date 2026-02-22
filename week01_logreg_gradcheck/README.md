# Week 01 - Logistic Regression + Gradient Checking (from scratch)

Week 1 demonstrates a full calculus-to-code loop for binary classification with labels in {-1, +1}.  
We derive logistic loss gradients from the chain rule and implement them from scratch with NumPy.  
We verify the analytic gradients using central-difference gradient checking before training.  
We then train with gradient descent on synthetic 2D data and visualize optimization behavior and the learned boundary.  
The result is a compact research-style artifact: derivation, implementation, verification, experiment, and tests.

## What's inside
- `derivation.md` - intuition + full gradient derivation
- `margin_plot.py` - plots logistic loss as a function of margin
- `experiment.py` - gradcheck + training + evaluation + plotting
- `src/utils.py` - dataset generation and stable sigmoid
- `src/logreg.py` - logistic loss/gradients + prediction
- `src/gradcheck.py` - finite-difference gradient checker
- `tests/test_utils.py` - utility tests
- `tests/test_logreg.py` - loss/gradient and gradcheck tests

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
python margin_plot.py
python experiment.py
python -m unittest -v
```

## Expected outputs
- `data_preview.png`
- `loss_vs_margin.png`
- `loss_curve.png`
- `decision_boundary.png`
