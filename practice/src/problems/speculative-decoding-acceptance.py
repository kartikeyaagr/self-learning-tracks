import numpy as np

META = dict(title='Speculative decoding: accept, reject and resample', tags=['Inference', 'Language Models', 'JAX'],
            diff='Hard', runtime='jax', lessons=['ieng:16.03', 'aieng:07.04'],
            related=['top-k-and-top-p-sampling', 'greedy-decode-loop-with-scan', 'beam-search-step'], tol=[1e-6, 1e-7])

STATEMENT = r"""
A small draft model proposes $K$ tokens; the large target model scores all of them in one forward pass; a rejection rule decides how many to keep, so the output has **exactly** the target model's distribution. Write the verification step for a batch, with the randomness passed in as uniforms so the result is deterministic.

Write `speculative_accept(draft, q, p, u, r)`:

- `draft`: `(B, K)` int, the drafted tokens $x_1 \dots x_K$.
- `q`: `(B, K, V)`, the draft model's distribution at each drafted position ($q_i(x_i) > 0$).
- `p`: `(B, K + 1, V)`, the target model's distributions at the same positions plus one extra position after the last draft.
- `u`: `(B, K)` and `r`: `(B,)`, uniforms in $[0, 1)$.

Per row:

1. Draft $i$ is **acceptable** if `u[i] * q[i, x_i] < p[i, x_i]` (that is, $u_i < p_i(x_i) / q_i(x_i)$). The number accepted, $n$, is the length of the leading run of acceptable drafts ($0 \le n \le K$).
2. If $n < K$, sample a replacement from the residual $\text{res} = \max(p_n - q_n, 0)$. If $n = K$, sample a bonus token from $\text{res} = p_K$.
3. Sample by inverse CDF with `r`: the token is the **first** index $j$ with $\mathrm{cumsum}(\text{res})_j > r \cdot \sum \text{res}$ (clip to $V - 1$).
4. The output row has length $K + 1$: drafts $x_1 \dots x_n$, then the sampled token, then `-1` padding.

Return `(n, out)`: `(B,)` int32 and `(B, K + 1)` int32. Write the logic for one row and `jax.vmap` it; no Python loop over $K$.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def speculative_accept(draft, q, p, u, r):
    """Return (accepted count (B,), output tokens (B, K + 1) padded with -1)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _one(draft, q, p, u, r):
    K, V = q.shape
    idx = jnp.arange(K)
    ok = u * q[idx, draft] < p[idx, draft]                      # (K,) acceptable drafts
    n = jnp.sum(jnp.cumprod(ok.astype(jnp.int32)))              # length of the leading run of Trues

    # Pad q with a zero row: at n = K the "residual" max(p_K - 0, 0) is just p_K, the bonus distribution.
    q_pad = jnp.concatenate([q, jnp.zeros((1, V), q.dtype)])
    res = jnp.maximum(p[n] - q_pad[n], 0.0)
    cdf = jnp.cumsum(res)
    tok = jnp.minimum(jnp.sum(cdf <= r * cdf[-1]), V - 1)      # first j with cdf[j] > r * total

    pos = jnp.arange(K + 1)
    padded = jnp.concatenate([draft, jnp.zeros(1, draft.dtype)])
    out = jnp.where(pos < n, padded, jnp.where(pos == n, tok, -1))
    return n.astype(jnp.int32), out.astype(jnp.int32)

def speculative_accept(draft, q, p, u, r):
    draft = jnp.asarray(draft, jnp.int32)
    q, p, u, r = (jnp.asarray(a, jnp.float32) for a in (q, p, u, r))
    return jax.vmap(_one)(draft, q, p, u, r)
'''

HINTS = [
    'Gather $q_i(x_i)$ and $p_i(x_i)$ with `q[jnp.arange(K), draft]`. The leading run of accepted drafts is `sum(cumprod(ok))`.',
    'Append a row of zeros to `q`. Then `max(p[n] - q_pad[n], 0)` covers both cases: the residual at the first rejection, and $p_K$ when everything was accepted.',
    'Inverse CDF: `jnp.sum(cdf <= r * cdf[-1])` counts the entries below the threshold, which is the index of the first one above it. Build the output with `jnp.where` on `arange(K + 1)` compared with `n`.',
]

_r = np.random.default_rng(33)

def _dist(*s, temp=1.0):
    z = _r.normal(size=s) / temp
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def _case(B, K, V, same=False):
    q = _dist(B, K, V)
    p = np.concatenate([q, _dist(B, 1, V)], axis=1) if same else _dist(B, K + 1, V)
    draft = np.array([[_r.choice(V, p=q[b, i]) for i in range(K)] for b in range(B)])
    return draft, q, p, _r.random((B, K)), _r.random(B)

_q1 = np.array([[[0.5, 0.5, 0.0], [0.2, 0.8, 0.0]]])
_p1 = np.array([[[0.5, 0.5, 0.0], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8]]])
TESTS = [
    (np.array([[1, 1]]), _q1, _p1, np.array([[0.9, 0.5]]), np.array([0.3])),
    (np.array([[1, 0]]), _q1, _p1, np.array([[0.1, 0.1]]), np.array([0.95])),
    _case(3, 4, 6),
    _case(2, 3, 5, same=True),
    _case(4, 5, 8),
    (np.array([[0, 2]]), np.array([[[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]]]),
     np.array([[[0.05, 0.9, 0.05], [0.3, 0.3, 0.4], [1.0, 0.0, 0.0]]]), np.array([[0.5, 0.0]]), np.array([0.99])),
    _case(6, 2, 4),
]

EXPLANATION = r"""
## Why the output distribution is exactly $p$
Accepting $x \sim q$ with probability $\min(1, p(x)/q(x))$ yields token $x$ with probability $\min(q(x), p(x))$. On rejection, sampling from the normalised residual $\max(p - q, 0)$ supplies the missing mass $p(x) - \min(q(x), p(x))$. The two add up to $p(x)$ for every $x$, whatever the draft model is. The draft only affects **speed**: the closer $q$ is to $p$, the more tokens are accepted per target pass.

## Walking through test 1
Draft 1 is token 1 with $q = p = 0.5$: ratio 1, accepted. Draft 2 is token 1 with $q = 0.8$, $p = 0.2$: ratio 0.25 and $u = 0.5$, so it is rejected and $n = 1$. The residual is $\max([0.6, 0.2, 0.2] - [0.2, 0.8, 0], 0) = [0.4, 0, 0.2]$; with $r = 0.3$ the threshold is 0.18, so token 0 is sampled. Output `[1, 0, -1]`. Test 4 uses $p = q$ on the drafted positions, so everything is accepted and a bonus token is drawn from $p_K$.

## Expected speed-up
If each draft is accepted with probability $\alpha$ independently, one verification yields $\frac{1 - \alpha^{K+1}}{1 - \alpha}$ tokens on average. The target pass over $K + 1$ positions costs about the same as one decode step because decode is memory-bound, which is where the speed-up comes from.

## Common bugs
- Counting all accepted drafts instead of the leading run: tokens after a rejection were conditioned on a rejected prefix.
- Sampling the replacement from $p_n$ instead of the residual: the output is then biased toward tokens the draft over-proposes.
- Forgetting the bonus token when all $K$ drafts are accepted: correct, but wastes a token per round.

## Interview follow-ups
- How does the rule change for greedy (temperature 0) decoding? (Accept while the draft equals the target argmax.)
- In a batch, rows accept different numbers of tokens. How does an engine keep the KV cache consistent afterwards?
"""
