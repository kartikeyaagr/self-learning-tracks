import numpy as np

META = dict(title='Skip-gram with negative sampling: one training step', tags=['Embeddings', 'NLP'], diff='Hard', runtime='numpy',
            lessons=['aieng:02.01'], related=['fancy-indexing', 'embedding-similarity', 'glove'])

STATEMENT = r"""
Word2vec's skip-gram model learns two tables: input vectors $v_w$ (rows of `W_in`) and output vectors $u_w$ (rows of `W_out`). For a center word $c$ and a context word $o$ it pushes $u_o \cdot v_c$ up, and for $K$ sampled "negative" words it pushes $u_k \cdot v_c$ down.

Write `sgns_step(tokens, W_in, W_out, negatives, window, lr)`.

## 1. Training pairs
For every position $i$ in `tokens` and every $j$ with $1 \le |i - j| \le$ `window` and $0 \le j <$ `len(tokens)`, form the pair (center `tokens[i]`, context `tokens[j]`). Order the pairs by $i$, then by $j$ ascending. Call their number $P$ (the tests always have $P \ge 1$).

## 2. Loss
`negatives` is an int array of shape `(P, K)`: row $p$ holds the negative word ids for pair $p$. With $\sigma(x) = 1/(1 + e^{-x})$:
$$L = \frac{1}{P} \sum_{p=1}^{P} \Big[ -\ln \sigma(u_{o_p} \cdot v_{c_p}) - \sum_{k=1}^{K} \ln \sigma(-u_{n_{pk}} \cdot v_{c_p}) \Big]$$

## 3. Update
Compute the gradient of $L$ with respect to **all** of `W_in` and `W_out` at the current parameters (a word used several times accumulates every contribution), then take one gradient-descent step: $W \leftarrow W - \text{lr} \cdot \nabla_W L$.

## Output
`(loss, new_W_in, new_W_out)`: `loss` is $L$ before the update as a float, and the two arrays have the input shapes. Do not modify the inputs.
"""

STARTER = '''import numpy as np

def sgns_step(tokens, W_in, W_out, negatives, window, lr):
    """Return (loss before the step, updated W_in, updated W_out)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)          # stable log(1 / (1 + e^-x))

def sgns_step(tokens, W_in, W_out, negatives, window, lr):
    W_in, W_out = np.asarray(W_in, dtype=float), np.asarray(W_out, dtype=float)
    T = len(tokens)
    pairs = [(tokens[i], tokens[j]) for i in range(T)
             for j in range(max(0, i - window), min(T, i + window + 1)) if j != i]
    c = np.array([a for a, _ in pairs])
    o = np.array([b for _, b in pairs])
    neg = np.asarray(negatives).reshape(len(pairs), -1)
    P = len(pairs)

    v = W_in[c]                                   # (P, d)
    s_pos = np.sum(W_out[o] * v, axis=1)          # (P,)
    s_neg = np.einsum("pkd,pd->pk", W_out[neg], v)  # (P, K)
    loss = -(log_sigmoid(s_pos).sum() + log_sigmoid(-s_neg).sum()) / P

    # dL/ds = (sigmoid(s) - 1)/P for positives and sigmoid(s)/P for negatives
    g_pos = (1.0 / (1.0 + np.exp(-s_pos)) - 1.0) / P
    g_neg = (1.0 / (1.0 + np.exp(-s_neg))) / P
    grad_v = g_pos[:, None] * W_out[o] + np.einsum("pk,pkd->pd", g_neg, W_out[neg])

    G_in, G_out = np.zeros_like(W_in), np.zeros_like(W_out)
    np.add.at(G_in, c, grad_v)                    # add.at accumulates repeated ids; G[c] += ... would not
    np.add.at(G_out, o, g_pos[:, None] * v)
    np.add.at(G_out, neg, g_neg[:, :, None] * v[:, None, :])
    return float(loss), W_in - lr * G_in, W_out - lr * G_out
'''

HINTS = [
    'Build the pair list with two nested loops, then gather everything at once: `v = W_in[centers]` is `(P, d)`, `W_out[negatives]` is `(P, K, d)`.',
    'With $s = u \\cdot v$, the derivative of $-\\ln\\sigma(s)$ is $\\sigma(s) - 1$ and the derivative of $-\\ln\\sigma(-s)$ is $\\sigma(s)$. Divide by $P$ for the mean. Then $\\partial/\\partial v = g\\,u$ and $\\partial/\\partial u = g\\,v$.',
    'Scatter the per-pair gradients back into full-size zero arrays with `np.add.at(G, ids, values)`. Plain `G[ids] += values` keeps only one contribution per repeated id. Compute every gradient from the old parameters before updating.',
]

_r = np.random.default_rng(8)
_V, _d = 10, 4
_Win, _Wout = _r.normal(scale=0.5, size=(_V, _d)), _r.normal(scale=0.5, size=(_V, _d))
_toks = [1, 4, 2, 4, 7, 1, 9, 3]
_npairs = lambda T, w: sum(min(T, i + w + 1) - max(0, i - w) - 1 for i in range(T))
TESTS = [
    ([0, 1], np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]), np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]),
     np.array([[2], [2]]), 1, 0.1),
    ([2, 0, 2], np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]), np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]),
     np.array([[1, 1], [1, 2], [0, 0], [1, 1]]), 1, 0.5),
    ([3, 3], np.zeros((4, 3)), np.zeros((4, 3)), np.array([[0], [1]]), 5, 1.0),
    ([0, 1, 2], np.eye(3), np.eye(3) * 2, np.array([[0], [1], [2], [0]]), 1, 0.0),
    (_toks, _Win, _Wout, _r.integers(0, _V, size=(_npairs(8, 2), 3)), 2, 0.05),
    (_toks[:5], _Win * 10, _Wout * 10, _r.integers(0, _V, size=(_npairs(5, 3), 5)), 3, 0.01),
]

EXPLANATION = r"""
## Why it works
A full softmax over the vocabulary would cost $O(Vd)$ per pair. Negative sampling replaces it with $K + 1$ binary logistic regressions: "is $o$ a real context of $c$?" (yes) and "is $n_k$?" (no). The gradient has a clean form: for a score $s$ with label $y$, $\partial \ell / \partial s = \sigma(s) - y$. Each pair therefore touches only one row of `W_in` and $K + 1$ rows of `W_out`, so a step costs $O(PKd)$ regardless of vocabulary size.

## Repeated indices
The same word shows up as a center many times, and a negative can repeat or even equal the context word. NumPy's `G[idx] += x` with repeated `idx` writes each element once (the last write wins), silently dropping gradient. `np.add.at` performs an unbuffered accumulation. Test 2 has repeated centers, contexts and negatives.

## Numerical stability
`np.logaddexp(0, -x)` computes $\ln(1 + e^{-x})$ without overflow, which matters in test 6 where the vectors are scaled by 10 and scores reach the hundreds.

## Complexity
$O(PKd)$ time and $O(PKd)$ temporary memory for the gathered negatives, plus $O(Vd)$ for the gradient arrays. Real implementations update rows in place, one pair at a time (asynchronous SGD), and never build dense gradients.

## Interview follow-ups
- Why are negatives drawn from the unigram distribution raised to the $3/4$ power? (It boosts rare words relative to frequency, which gave better vectors empirically.)
- What does subsampling frequent words do, and how does it change the effective window?
"""
