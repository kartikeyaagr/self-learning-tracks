import numpy as np

META = dict(title='Chunkwise gated delta rule with the WY/UT transform', tags=['Language Models', 'Attention'], diff='Hard',
            runtime='numpy', lessons=['aieng:07.02'],
            related=['gated-deltanet-state-update', 'parallel-and-recurrent-linear-attention', 'mamba-2-gated-state-scan'])

STATEMENT = r"""
The gated delta rule $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, $o_t = S_t q_t$ is a sequential recurrence. To train on GPUs it is computed **chunk by chunk**: inside a chunk all tokens are handled with matrix products, and only the state crosses chunk boundaries.

Within a chunk with incoming state $S_0$, let $\gamma_t = \prod_{s \le t} \alpha_s$ (cumulative inside the chunk, $t = 1..C$). The state can be written in WY form
$$S_t = \gamma_t S_0 + \sum_{i \le t} \frac{\gamma_t}{\gamma_i}\, u_i k_i^\top$$
where the "pseudo-values" $u_i$ solve a unit lower-triangular system:
$$(I + A)\,U = \mathrm{diag}(\beta)\,\big(V - \mathrm{diag}(\gamma)\,K S_0^\top\big),\quad A_{t,i} = \beta_t\,\frac{\gamma_t}{\gamma_i}\,k_t^\top k_i \ \ (i < t),\ 0 \text{ otherwise}.$$
Then $o_t = \gamma_t S_0 q_t + \sum_{i \le t} \frac{\gamma_t}{\gamma_i}(k_i^\top q_t)\,u_i$, and the outgoing state is $S_C$.

Write `chunked_gated_delta(q, k, v, alpha, beta, chunk)` for **one head** returning `(o, S_T)`.
- `q`, `k`: `(T, d_k)` (already normalised and scaled, use as given); `v`: `(T, d_v)`; `alpha`, `beta`: `(T,)`. $S_0 = 0$ at the start.
- Chunks are consecutive blocks of `chunk` tokens; the last one may be shorter.
- `o`: `(T, d_v)`; `S_T`: `(d_v, d_k)`. Results must equal the token-by-token recurrence.
"""

STARTER = '''import numpy as np

def chunked_gated_delta(q, k, v, alpha, beta, chunk):
    """Return (o (T, dv), S_T (dv, dk))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def chunked_gated_delta(q, k, v, alpha, beta, chunk):
    q, k, v = (np.asarray(a, dtype=float) for a in (q, k, v))
    alpha, beta = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    T, dk = k.shape
    S = np.zeros((v.shape[1], dk))
    o = np.empty((T, v.shape[1]))
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        Q, K, V, be = q[a:b], k[a:b], v[a:b], beta[a:b]
        g = np.cumprod(alpha[a:b])                              # gamma_t inside the chunk
        ratio = np.tril(g[:, None] / g[None, :])                # gamma_t / gamma_i for i <= t
        A = np.tril(be[:, None] * ratio * (K @ K.T), k=-1)      # strictly lower
        rhs = be[:, None] * (V - g[:, None] * (K @ S.T))
        U = np.linalg.solve(np.eye(b - a) + A, rhs)             # unit lower-triangular solve
        o[a:b] = g[:, None] * (Q @ S.T) + (ratio * (Q @ K.T)) @ U
        S = g[-1] * S + (U * (g[-1] / g)[:, None]).T @ K
    return o, S
'''

HINTS = [
    'Inside a chunk build three small matrices: the decay ratios $\\gamma_t/\\gamma_i$ (lower triangle), $KK^\\top$ and $QK^\\top$. Everything else is products of these with $U$ and $S_0$.',
    'The right-hand side subtracts what the incoming state already predicts for each key, decayed to that position: $\\gamma_t S_0 k_t$. Solve $(I + A)U = \\text{rhs}$; since $I + A$ is unit lower-triangular, forward substitution works too.',
    'Outgoing state: $S_C = \\gamma_C S_0 + \\sum_i (\\gamma_C/\\gamma_i) u_i k_i^\\top$, i.e. `g[-1] * S + (U * (g[-1] / g)[:, None]).T @ K`.',
]

_r = np.random.default_rng(2412)
def _case(T, dk, dv, lo=0.7):
    k = _r.normal(size=(T, dk)); k /= np.linalg.norm(k, axis=1, keepdims=True)
    q = _r.normal(size=(T, dk)); q /= np.linalg.norm(q, axis=1, keepdims=True) * np.sqrt(dk)
    return q, k, _r.normal(size=(T, dv)), _r.uniform(lo, 1.0, size=T), _r.uniform(0.0, 1.0, size=T)
_c1, _c2, _c3 = _case(6, 3, 2), _case(10, 4, 3), _case(9, 4, 4, 0.9)
TESTS = [
    (*_c1, 6),
    (*_c1, 2),
    (*_c1, 1),
    (*_c2, 4),
    (*_c2, 3),
    (*_c3, 5),
    (*_case(7, 2, 2, 1.0), 3),
]

EXPLANATION = r"""
## Where the WY form comes from
Expand the recurrence inside a chunk and every update is "decay, then add a rank-1 term". Collecting the rank-1 terms gives $S_t = \gamma_t S_0 + \sum_i (\gamma_t/\gamma_i) u_i k_i^\top$, and matching the delta-rule correction term by term shows that $u_t = \beta_t(v_t - \text{what } S_{t-1} \text{ predicts for } k_t)$. Those predictions depend on earlier $u_i$, which is exactly the triangular system. ($u_t$ equals $\beta_t\delta_t$ from the recurrence, so the chunked and sequential versions share their intermediate quantities, not just their outputs.)

## Why this is fast
The sequential version does $T$ small rank-1 updates. The chunked version replaces them with $T/C$ batches of $C\times C$ and $C\times d$ matrix products plus one small triangular solve per chunk, which maps onto tensor cores. Gated DeltaNet's kernels compute $(I + A)^{-1}$ with a "UT transform" (a few matrix products) rather than a general solve.

## Numerics
$\gamma_t/\gamma_i$ is computed as a ratio of cumulative products, which underflows for long chunks with strong decay. Real kernels work in log space ($\sum \log\alpha$) and keep chunks short (64 tokens).

## Complexity
Per chunk $O(C^2(d_k + d_v) + C^3 + C d_k d_v)$; overall $O(TC(d_k + d_v) + TC^2 + T d_k d_v)$, linear in $T$ for a fixed chunk size.

## Common bugs
- Including the diagonal in $A$ (it must be unit lower-triangular, the diagonal is the identity).
- Forgetting to decay the incoming state's prediction by $\gamma_t$.
- Restarting $\gamma$ at 1 inside the chunk but also multiplying by the previous chunk's decay (it is already in $S_0$).

## Interview follow-ups
- What changes if $\alpha_t = 1$ everywhere (plain DeltaNet)?
- How does sequence parallelism across devices work with this chunk structure?
"""
