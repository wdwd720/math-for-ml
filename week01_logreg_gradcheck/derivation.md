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

## Interpretation (what the model is doing)
We compute a linear score:
z = w · x + b
- If z is large positive → class +1
- If z is large negative → class −1
Decision boundary: z = 0

Probabilistic view:
P(y=+1 | x) = σ(z), σ(t)=1/(1+e^(−t))
(We keep y ∈ {−1,+1} for cleaner algebra.)

## Margin viewpoint (AIME-style insight)
Define margin m = y z = y(w·x + b)
- m >> 0: correct & confident
- m ≈ 0: uncertain
- m < 0: wrong
Logistic loss as function of margin:
ℓ(m)=log(1+exp(−m))
Training tries to increase margin.

## Loss values (intuition check)
Add a table:

| margin m | loss ℓ(m) (approx) |
|---:|---:|
| -3 | 3.048 |
| -1 | 1.313 |
|  0 | 0.693 |
|  1 | 0.313 |
|  3 | 0.049 |

Large negative margins are penalized heavily, while large positive margins incur near-zero loss.
