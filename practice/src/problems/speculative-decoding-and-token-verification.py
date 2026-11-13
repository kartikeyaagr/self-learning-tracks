import numpy as np

META = dict(title='Speculative decoding: verify draft tokens by rejection sampling', tags=['Inference', 'Language Models', 'Sampling'],
            diff='Hard', runtime='numpy', lessons=['ieng:16.03', 'ieng:16.02'],
            related=['text-generation-sampling', 'temperature-top-k-and-top-p-sampling', 'autoregressive-decoding-with-kv-cache'])

STATEMENT = r"""
In speculative decoding a small **draft** model proposes $k$ tokens, then the large **target** model scores all of them in one forward pass. A rejection-sampling rule decides how many drafts to keep, and the result is distributed **exactly** as if the target had sampled every token itself.

Write `verify(draft, q, p, u_accept, u_sample)`.

## Inputs
- `draft`: int array `(k,)`, $k \ge 1$, the proposed tokens $x_0, \dots, x_{k-1}$.
- `q`: float `(k, V)`: row $i$ is the draft model's distribution that $x_i$ was sampled from (so $q_i(x_i) > 0$).
- `p`: float `(k + 1, V)`: row $i$ is the target's distribution at the same position; row $k$ is the target's distribution after all $k$ drafts.
- `u_accept`: float `(k,)` and `u_sample`: float, uniform draws in $[0, 1)$.

## Rule
For $i = 0, 1, \dots$:
1. **Accept** $x_i$ if `u_accept[i]` $< p_i(x_i) / q_i(x_i)$ (always true when $p_i(x_i) \ge q_i(x_i)$). Append it and continue.
2. Otherwise **reject**: stop, and sample one replacement token from the residual distribution $r = \max(p_i - q_i, 0) / \sum \max(p_i - q_i, 0)$.

If all $k$ drafts are accepted, sample one **bonus** token from $p_k$. Either way exactly one sampled token is appended, so a call produces between $1$ and $k + 1$ tokens.

**Sampling from a distribution $w$ with `u_sample`:** the smallest index $j$ with $\sum_{l \le j} w_l >$ `u_sample`; if rounding leaves none, the largest index with $w_j > 0$.

## Output
A tuple `(tokens, n_accepted, alpha)`:
- `tokens`: list of Python ints, the accepted drafts followed by the sampled token.
- `n_accepted`: int, the number of drafts accepted.
- `alpha`: float array `(k,)`, the acceptance probability of each position, $\alpha_i = \sum_x \min(p_i(x), q_i(x))$ (the chance a fresh draft at position $i$ survives).
"""

STARTER = '''import numpy as np

def verify(draft, q, p, u_accept, u_sample):
    """Return (tokens list, n_accepted int, alpha (k,) array)."""
    k = len(draft)
    return [int(t) for t in draft], k, np.ones(k)       # accepts everything, no bonus token
'''

SOLUTION = '''import numpy as np

def _sample(w, u):
    c = np.cumsum(w)
    hit = np.nonzero(c > u)[0]
    return int(hit[0]) if len(hit) else int(np.nonzero(w > 0)[0][-1])

def verify(draft, q, p, u_accept, u_sample):
    k = len(draft)
    alpha = np.minimum(p[:k], q).sum(axis=1)       # overlap of the two distributions per position
    tokens = []
    for i in range(k):
        x = int(draft[i])
        if u_accept[i] < p[i, x] / q[i, x]:         # accept with probability min(1, p/q)
            tokens.append(x)
            continue
        resid = np.maximum(p[i] - q[i], 0.0)        # mass the target wants but the draft under-proposed
        tokens.append(_sample(resid / resid.sum(), u_sample))
        return tokens, i, alpha
    tokens.append(_sample(p[k], u_sample))          # every draft accepted: free bonus token from the target
    return tokens, k, alpha
'''

HINTS = [
    'Walk the drafts in order and stop at the first rejection. Accepting when `u < p/q` is the same as accepting with probability $\\min(1, p/q)$.',
    'On rejection the replacement comes from $\\max(p_i - q_i, 0)$, renormalised: exactly the probability mass the draft under-proposed. It is never all zero, because a rejection implies $q_i(x_i) > p_i(x_i)$.',
    'If no draft is rejected, the target\'s last row $p_k$ is still unused: sample the bonus token from it. `alpha` is just `np.minimum(p[:k], q).sum(axis=1)`.',
]


def _dists(rng, n, V, conc):
    return rng.dirichlet(np.full(V, conc), size=n)


def _case(seed, k, V, mix):
    rng = np.random.default_rng(seed)
    p = _dists(rng, k + 1, V, 0.7)
    q = mix * p[:k] + (1 - mix) * _dists(rng, k, V, 0.7)        # draft = a blend toward the target
    draft = np.array([rng.choice(V, p=q[i]) for i in range(k)])
    return (draft, q, p, rng.uniform(size=k), float(rng.uniform()))


_p2 = np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
TESTS = [
    (np.array([0, 2]), np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]), _p2, np.array([0.9, 0.9]), 0.75),
    (np.array([1, 0]), np.array([[0.2, 0.8, 0.0], [0.6, 0.2, 0.2]]), _p2, np.array([0.5, 0.1]), 0.3),
    (np.array([1, 0]), np.array([[0.2, 0.8, 0.0], [0.6, 0.2, 0.2]]), _p2, np.array([0.7, 0.1]), 0.3),
    (np.array([2]), np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 1.0], [0.25, 0.25, 0.5]]), np.array([0.999]), 0.999999),
    _case(1, 4, 6, 0.8),
    _case(14, 5, 8, 0.3),
    _case(7, 5, 8, 0.3),
    _case(13, 3, 5, 0.95),
    _case(4, 6, 10, 0.0),
]

EXPLANATION = r"""
## Why it works
For one position, the output token is $x$ with probability
$$q(x)\min\!\Bigl(1, \tfrac{p(x)}{q(x)}\Bigr) + \Bigl(1 - \sum_y \min(p(y), q(y))\Bigr)\,\frac{\max(p(x) - q(x), 0)}{\sum_y \max(p(y) - q(y), 0)}.$$
The first term is $\min(p, q)$. The two normalisers are both $1 - \alpha$, so the second term is $\max(p - q, 0)$. Their sum is $p(x)$: the output is an exact sample from the target, whatever the draft model is. A bad draft only lowers the acceptance rate, never the quality.

## Speed
With per-position acceptance $\alpha$ (assumed constant) the expected number of tokens per target pass is $(1 - \alpha^{k+1}) / (1 - \alpha)$. With $\alpha = 0.8$ and $k = 4$ that is about 3.4 tokens for one target pass plus $k$ cheap draft passes. Decode is memory-bound, so verifying $k + 1$ positions costs roughly the same time as generating one: the win is large at low batch sizes and shrinks as the batch grows and the target becomes compute-bound.

## Common bugs
- Resampling from $p_i$ instead of the residual on rejection: the output is then biased toward tokens the draft likes.
- Forgetting the bonus token when every draft is accepted, which wastes the target's last row.
- Continuing after a rejection: later drafts were conditioned on a token that is no longer in the sequence.

## Interview follow-ups
- With greedy decoding, what does the rule reduce to? (Accept while the draft token equals the target argmax; replace the first mismatch with the target argmax.)
- How do you pick $k$? (Balance draft cost against expected accepted length; adaptive schemes shrink $k$ when recent acceptance is low or the batch is large.)
"""
