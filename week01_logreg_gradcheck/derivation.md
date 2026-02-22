# Derivation Scaffold (Week 01)

## Model definitions
- \(x \in \mathbb{R}^d\)
- \(w \in \mathbb{R}^d\)
- \(b \in \mathbb{R}\)
- \(z = w \cdot x + b\)
- \(y \in \{-1, +1\}\)

## Logistic loss
\[
\ell(w,b;x,y) = \log\left(1 + \exp(-y z)\right)
\]

## Goal
Derive \(\nabla_w \ell\) and \(\partial \ell / \partial b\) later this week.
