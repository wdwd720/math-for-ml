# Derivation Scaffold (Week 01)

## Model definitions
- $x \in \mathbb{R}^d$
- $w \in \mathbb{R}^d$
- $b \in \mathbb{R}$
- $z = w \cdot x + b$
- $y \in \{-1, +1\}$

## Logistic loss
```math
\ell(w,b;x,y) = \log\left(1 + \exp(-y z)\right)
```

## Goal
Derive $\nabla_w \ell$ and $\partial \ell / \partial b$.

## Interpretation (what the model is doing)
We compute a linear score:
$z = w \cdot x + b$
- If $z$ is large positive -> class $+1$
- If $z$ is large negative -> class $-1$
Decision boundary: $z = 0$

Probabilistic view:
$P(y=+1 \mid x) = \sigma(z),\; \sigma(t)=\frac{1}{1+e^{-t}}$
($y \in \{-1,+1\}$ for cleaner algebra.)

## Margin viewpoint (AIME-style insight)
Define margin $m = y z = y(w \cdot x + b)$
- $m \gg 0$: correct and confident
- $m \approx 0$: uncertain
- $m < 0$: wrong
Logistic loss as function of margin:
$\ell(m)=\log(1+\exp(-m))$
Training tries to increase margin.

## Loss values (intuition check)
Add a table:

| margin $m$ | loss $\ell(m)$ (approx) |
|---:|---:|
| -3 | 3.048 |
| -1 | 1.313 |
|  0 | 0.693 |
|  1 | 0.313 |
|  3 | 0.049 |

Large negative margins are penalized heavily, while large positive margins incur near-zero loss.

## Full gradient derivation
For one sample $(x, y)$, define:
```math
z = w \cdot x + b,\quad u = -y z,\quad \ell = \log(1+\exp(u)) = \mathrm{logaddexp}(0,u).
```

First differentiate the outer function:
```math
\frac{d}{du}\log(1+\exp(u))
= \frac{\exp(u)}{1+\exp(u)}
= \sigma(u).
```

Chain rule terms:
```math
\frac{\partial u}{\partial w} = -y x,\qquad \frac{\partial u}{\partial b} = -y.
```

Therefore:
```math
\nabla_w \ell
= \frac{d\ell}{du}\frac{\partial u}{\partial w}
= \sigma(u)(-y x),
```
```math
\frac{\partial \ell}{\partial b}
= \frac{d\ell}{du}\frac{\partial u}{\partial b}
= \sigma(u)(-y).
```

For a dataset $\{(x_i, y_i)\}_{i=1}^n$, with
```math
L(w,b) = \frac{1}{n}\sum_{i=1}^{n}\ell_i,\qquad
u_i = -y_i(w \cdot x_i + b),
```
the averaged gradients are:
```math
\nabla_w L = \frac{1}{n}\sum_{i=1}^{n}\sigma(u_i)(-y_i x_i),\qquad
\frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}\sigma(u_i)(-y_i).
```
