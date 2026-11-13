import numpy as np

META = dict(title='Jacobian of softmax', tags=['Calculus', 'Activation Functions'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.08', 'mlmaths:09.04'], related=['softmax-rows', 'cross-entropy-logits'])

STATEMENT = r"""
For $s = \mathrm{softmax}(z)$ with $z \in \mathbb{R}^n$, the Jacobian $J \in \mathbb{R}^{n \times n}$ has entries $J_{ij} = \partial s_i / \partial z_j$.

Write `softmax_jacobian(z)` that returns $J$ for a 1-D input.

## Requirements
- Input: `z`, shape `(n,)`. Output: shape `(n, n)`.
- Compute softmax stably (subtract the max) first.
- No Python loops: build $J$ from $s$ with vector operations.
"""

STARTER = '''import numpy as np

def softmax_jacobian(z):
    """z: shape (n,). Return J with J[i, j] = d softmax(z)_i / d z_j, shape (n, n)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def softmax_jacobian(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    s = e / e.sum()
    return np.diag(s) - np.outer(s, s)      # J_ij = s_i (delta_ij - s_j)
'''

HINTS = [
    'Differentiate $s_i = e^{z_i}/\\sum_k e^{z_k}$ with the quotient rule. Treat the cases $i = j$ and $i \\ne j$ separately.',
    'You should get $J_{ii} = s_i(1 - s_i)$ and $J_{ij} = -s_i s_j$ for $i \\ne j$.',
    'Both cases are $s_i(\\delta_{ij} - s_j)$, which is `np.diag(s) - np.outer(s, s)`.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([0.0, 0.0]),),
    (np.array([1.0, 2.0, 3.0]),),
    (np.array([5.0]),),
    (np.array([100.0, 100.0, -100.0]),),
    (_r.normal(size=6),),
    (_r.normal(size=4) * 5,),
]

EXPLANATION = r"""
## Derivation
$s_i = e^{z_i}/S$ with $S = \sum_k e^{z_k}$. Then $\partial s_i/\partial z_j = \frac{\delta_{ij} e^{z_i} S - e^{z_i} e^{z_j}}{S^2} = s_i(\delta_{ij} - s_j)$. In matrix form, $J = \mathrm{diag}(s) - s s^\top$.

## Properties worth knowing
- $J$ is symmetric, and every row sums to 0. Shifting all logits by a constant doesn't change $s$, so moving along $\mathbf{1}$ has zero derivative.
- $J$ is singular: $J\mathbf{1} = 0$.

## Why you rarely build it
In training, softmax is followed by cross-entropy, and $\partial L / \partial z = s - y$ directly, which costs $O(n)$ instead of $O(n^2)$. Frameworks fuse the two operations for exactly this reason. The full Jacobian matters for analysis and for vector-Jacobian products where the upstream gradient isn't one-hot: $J^\top g = s \odot (g - s^\top g)$.

## Interview follow-ups
- Derive the gradient of cross-entropy with respect to the logits using this Jacobian.
- What is the Jacobian of `sigmoid`? (It's diagonal: elementwise $\sigma(1-\sigma)$.)
"""
