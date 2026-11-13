import numpy as np

META = dict(title='Exact speculative sampling with residual resampling', tags=['Inference', 'Sampling'], diff='Hard',
            runtime='numpy', lessons=['ieng:16.03', 'aieng:07.04'],
            related=['text-generation-sampling', 'decode-attention-with-kv-cache'])

STATEMENT = r"""
Speculative decoding lets a cheap **draft** model propose $k$ tokens and a large **target** model check them all in one forward pass. With the right accept/reject rule, the tokens that come out are distributed **exactly** as if the target model had sampled them one at a time.

Write `speculative_sample(draft_tokens, q, p, u_accept, u_sample)` and return a dict.

## Inputs
- `draft_tokens`: int array `(k,)`, $k \ge 1$, the tokens the draft model sampled.
- `q`: float64 `(k, V)`. Row $i$ is the draft model's distribution at draft position $i$ (the one `draft_tokens[i]` was sampled from), so `q[i, draft_tokens[i]] > 0`.
- `p`: float64 `(k + 1, V)`. Row $i < k$ is the target model's distribution at the same position; row $k$ is the target's distribution for the token **after** all $k$ drafts.
- `u_accept`: float64 `(k,)` and `u_sample`: float, uniform numbers in $[0, 1)$ that replace all randomness.

## Rule
Walk $i = 0, 1, \dots, k-1$ with $x = $ `draft_tokens[i]`:
1. Its acceptance probability is $a_i = \min\!\big(1,\ p_{i,x} / q_{i,x}\big)$. **Accept** if `u_accept[i] < a_i`, append $x$, and continue.
2. Otherwise **reject**: sample one token from the residual distribution $r \propto \max(p_i - q_i, 0)$ (elementwise over the vocabulary), append it, and stop.

If all $k$ drafts are accepted, sample one **bonus** token from $p_k$ and append it. Either way exactly one token is sampled, and it uses `u_sample`.

## Sampling convention
To sample from non-negative weights $w$ (not necessarily normalised) with total $W$, return the smallest index $j$ with $\sum_{t \le j} w_t > u \cdot W$ (inverse CDF). Tests keep `u_sample` away from bucket boundaries.

## Output
| key | value |
|---|---|
| `"tokens"` | list of Python ints: accepted drafts, then the sampled token (length $n_\text{acc} + 1$) |
| `"n_accepted"` | int |
| `"accept_probs"` | float64 `(k,)`, $a_i$ for **every** draft position, including those after a rejection |
"""

STARTER = '''import numpy as np

def speculative_sample(draft_tokens, q, p, u_accept, u_sample):
    """Return {"tokens": [...], "n_accepted": int, "accept_probs": (k,) array}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _inverse_cdf(w, u):
    """Smallest j with cumsum(w)[j] > u * sum(w)."""
    c = np.cumsum(w)
    return int(np.searchsorted(c, u * c[-1], side="right"))

def speculative_sample(draft_tokens, q, p, u_accept, u_sample):
    x = np.asarray(draft_tokens)
    q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
    k = len(x)
    rows = np.arange(k)
    accept_probs = np.minimum(1.0, p[rows, x] / q[rows, x])

    tokens = []
    for i in range(k):
        if u_accept[i] < accept_probs[i]:
            tokens.append(int(x[i]))
            continue
        # Rejected: the residual covers exactly the mass the draft under-proposed.
        residual = np.maximum(p[i] - q[i], 0.0)
        tokens.append(_inverse_cdf(residual, u_sample))
        return {"tokens": tokens, "n_accepted": i, "accept_probs": accept_probs}
    tokens.append(_inverse_cdf(p[k], u_sample))         # all accepted: free bonus token from the target
    return {"tokens": tokens, "n_accepted": k, "accept_probs": accept_probs}
'''

HINTS = [
    'Compute every $a_i$ up front with fancy indexing `p[rows, draft_tokens] / q[rows, draft_tokens]`, then walk the positions in a loop that stops at the first rejection.',
    'On rejection the replacement comes from `np.maximum(p[i] - q[i], 0)`, not from `p[i]`: sampling from $p$ again would over-represent tokens the draft already proposed often.',
    'For the inverse CDF, `np.searchsorted(np.cumsum(w), u * w.sum(), side="right")` returns the smallest $j$ whose cumulative weight exceeds $uW$. If nothing is rejected, sample the bonus token from `p[k]`.',
]


def _dist(r, n, V):
    z = r.normal(size=(n, V)) * 1.5
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


_r = np.random.default_rng(11)
_q = _dist(_r, 5, 8)
_p = _dist(_r, 6, 8)
_x = np.array([int(np.argmax(row)) for row in _q])
TESTS = [
    (np.array([0]), np.array([[0.5, 0.5]]), np.array([[0.25, 0.75], [0.9, 0.1]]), np.array([0.3]), 0.5),
    (np.array([0, 1]), np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.6, 0.4], [0.2, 0.8], [0.3, 0.7]]),
     np.array([0.9, 0.1]), 0.2),
    (np.array([1, 2]), np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
     np.array([[0.2, 0.2, 0.6], [0.3, 0.3, 0.4], [1 / 3, 1 / 3, 1 / 3]]), np.array([0.39, 0.0]), 0.7),
    (np.array([1, 2]), np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
     np.array([[0.2, 0.2, 0.6], [0.3, 0.3, 0.4], [1 / 3, 1 / 3, 1 / 3]]), np.array([0.41, 0.0]), 0.7),
    (_x, _q, _p, np.array([0.05, 0.1, 0.02, 0.07, 0.03]), 0.4321),
    (_x, _q, _p, np.array([0.01, 0.97, 0.5, 0.5, 0.5]), 0.123),
    (np.array([3, 3, 3]), np.tile(np.array([0.1, 0.2, 0.3, 0.4]), (3, 1)), np.tile(np.array([0.1, 0.2, 0.3, 0.4]), (4, 1)),
     np.array([0.99, 0.99, 0.99]), 0.95),
    (np.array([2]), np.array([[0.0, 0.0, 1.0]]), np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]), np.array([0.0]), 0.75),
]

EXPLANATION = r"""
## Why the output is exactly the target distribution
At one position, the probability of emitting token $y$ is the chance the draft proposes $y$ and it survives, plus the chance of a rejection followed by resampling $y$:

$$q(y)\min\!\Big(1, \tfrac{p(y)}{q(y)}\Big) + \Big(1 - \sum_z \min(p(z), q(z))\Big) \frac{\max(p(y) - q(y), 0)}{\sum_z \max(p(z) - q(z), 0)}$$

The first term is $\min(p(y), q(y))$. The two normalisers are equal (both are the total variation distance between $p$ and $q$), so the second term is $\max(p(y) - q(y), 0)$, and the sum is $p(y)$. Accepted prefixes keep the context identical to what the target would have seen, so the argument applies position by position.

## Speed
One target forward pass verifies $k$ drafts and always yields at least one token, so the worst case is no slower than normal decoding (apart from the draft's cost). With per-token acceptance rate $\alpha$, the expected number of tokens per target pass is $(1 - \alpha^{k+1}) / (1 - \alpha)$. Tests 3 and 4 differ only in `u_accept[0]` around $a_0 = 0.4$: one rejects immediately, the other accepts both drafts and takes a bonus token. Test 7 is the case $p = q$, where every draft is accepted regardless of the uniforms.

## Common bugs
- Resampling from $p_i$ instead of the residual. The output is then biased toward tokens the draft likes.
- Forgetting the bonus token when everything is accepted (you lose a free token per pass).
- Using `<=` in the accept test. With continuous uniforms it rarely matters, but when $a_i = 0$ (test 8), `u = 0.0` must still reject.
- Comparing $u$ to $p/q$ without the `min(1, ·)` and then reporting probabilities above 1.

## Interview follow-ups
- With greedy decoding (temperature 0), what does this rule reduce to?
- How do tree-based drafts (several candidate continuations per position) change the acceptance rule?
"""
