import numpy as np

META = dict(title='Gated delta-rule state updates', tags=['Language Models', 'Attention'], diff='Hard', runtime='numpy',
            lessons=['aieng:07.02', 'air-fanout-v1:08.04'],
            related=['parallel-and-recurrent-linear-attention', 'mamba-2-gated-state-scan', 'scaled-dot-product-attention'])

STATEMENT = r"""
Linear attention only ever **adds** to its state $S$, so old associations pile up and interfere. The delta rule instead treats $S$ as a key-to-value memory and corrects it: read what $S$ currently returns for key $k_t$, and move it toward $v_t$ by a step $\beta_t$. A gate $\alpha_t$ additionally decays the whole memory, letting the model forget.

Write `gated_delta(q, k, v, alpha, beta, S0=None)`.

## Inputs (all `float64`)
- `q`, `k`: `(T, H, d_k)`; `v`: `(T, H, d_v)`.
- `alpha`: `(T, H)` with values in $(0, 1]$ (decay); `beta`: `(T, H)` in $[0, 1]$ (write strength).
- `S0`: `None` (zeros) or `(H, d_v, d_k)`, the memory before the first token.

## Per head, for $t = 1, \dots, T$
1. Normalise: $\hat k_t = k_t / \max(\lVert k_t \rVert_2, 10^{-6})$ and $\hat q_t = q_t / \max(\lVert q_t \rVert_2, 10^{-6})$, then scale the query: $\tilde q_t = \hat q_t / \sqrt{d_k}$.
2. Decay: $S' = \alpha_t S_{t-1}$.
3. Read and correct: $\text{old}_t = S' \hat k_t$, $\;\delta_t = v_t - \text{old}_t$, $\;S_t = S' + \beta_t\, \delta_t \hat k_t^\top$.
4. Output: $o_t = S_t \tilde q_t$.

Steps 2 and 3 together are $S_t = \alpha_t S_{t-1}(I - \beta_t \hat k_t \hat k_t^\top) + \beta_t v_t \hat k_t^\top$.

## Output
A tuple `(o, S_T, err)`:
- `o`: `(T, H, d_v)`; `S_T`: `(H, d_v, d_k)`;
- `err`: `(T, H)`, the prediction error $\lVert \delta_t \rVert_2$ before each write (how "surprised" the memory was).

Do not modify the inputs.
"""

STARTER = '''import numpy as np

def gated_delta(q, k, v, alpha, beta, S0=None):
    """Return (o (T, H, dv), S_T (H, dv, dk), err (T, H))."""
    T, H, dk = k.shape
    S = np.zeros((H, v.shape[2], dk)) if S0 is None else S0
    o = np.zeros((T, H, v.shape[2]))
    err = np.zeros((T, H))
    for t in range(T):
        S = S + beta[t][:, None, None] * v[t][:, :, None] * k[t][:, None, :]   # TODO: plain linear attention
        o[t] = np.einsum('hvk,hk->hv', S, q[t])
    return o, S, err
'''

SOLUTION = '''import numpy as np

def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)

def gated_delta(q, k, v, alpha, beta, S0=None):
    T, H, dk = k.shape
    dv = v.shape[2]
    kh = _unit(k)
    qh = _unit(q) / np.sqrt(dk)
    S = np.zeros((H, dv, dk)) if S0 is None else S0.copy()
    o = np.empty((T, H, dv))
    err = np.empty((T, H))
    for t in range(T):
        S = alpha[t][:, None, None] * S                          # forget a little
        old = np.einsum('hvk,hk->hv', S, kh[t])                  # what the memory says for this key
        delta = v[t] - old
        err[t] = np.linalg.norm(delta, axis=-1)
        S = S + beta[t][:, None, None] * delta[:, :, None] * kh[t][:, None, :]   # rank-1 correction
        o[t] = np.einsum('hvk,hk->hv', S, qh[t])                 # read with the updated memory
    return o, S, err
'''

HINTS = [
    'Normalise the keys first: with $\\lVert \\hat k \\rVert = 1$ and $\\beta = 1$, one write makes $S\\hat k = v$ exactly (test 1).',
    'Decay **before** reading: `old = (alpha * S) @ k_hat`, `delta = v - old`. The error output is the norm of this `delta`.',
    'Batch over heads with `einsum("hvk,hk->hv", S, x)` for the reads and `delta[:, :, None] * k_hat[:, None, :]` for the rank-1 writes. Read the output after the write.',
]

_r = np.random.default_rng(13)


def _case(T, H, dk, dv, s0=False):
    return (_r.normal(size=(T, H, dk)), _r.normal(size=(T, H, dk)), _r.normal(size=(T, H, dv)),
            _r.uniform(0.5, 1.0, size=(T, H)), _r.uniform(0.0, 1.0, size=(T, H)),
            *((_r.normal(size=(H, dv, dk)),) if s0 else ()))


# test 3: orthogonal keys, beta = 1, alpha = 1: the memory stores both pairs exactly, then overwrites the first
_e = np.eye(2)
TESTS = [
    (np.array([[[2.0, 0.0]]]), np.array([[[3.0, 0.0]]]), np.array([[[5.0]]]), np.ones((1, 1)), np.ones((1, 1))),
    (np.array([[[1.0, 0.0]], [[1.0, 0.0]]]), np.array([[[1.0, 0.0]], [[1.0, 0.0]]]), np.array([[[4.0]], [[4.0]]]),
     np.ones((2, 1)), np.full((2, 1), 0.5)),
    (np.stack([_e[0], _e[1], _e[0]])[:, None], np.stack([_e[0], _e[1], _e[0]])[:, None],
     np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[-1.0, 0.0]]]), np.ones((3, 1)), np.ones((3, 1))),
    _case(6, 2, 4, 3),
    _case(5, 3, 3, 2, True),
    (np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), np.ones((2, 1, 2)), np.full((2, 1), 0.9), np.ones((2, 1))),
    _case(12, 2, 8, 4),
    (*_case(4, 1, 2, 2)[:3], np.full((4, 1), 0.5), np.zeros((4, 1)), np.ones((1, 2, 2))),
]

EXPLANATION = r"""
## Why it works
The delta rule is one step of online gradient descent on the regression loss $\tfrac12 \lVert S\hat k_t - v_t \rVert^2$: its gradient is $-(v_t - S\hat k_t)\hat k_t^\top$, so $S \leftarrow S + \beta_t \delta_t \hat k_t^\top$ is SGD with learning rate $\beta_t$. With unit keys and $\beta = 1$ it overwrites the value for $\hat k_t$ exactly while leaving orthogonal directions untouched (test 3: the third token replaces the first association and keeps the second). Plain linear attention would instead have stored the sum $v_1 + v_3$ for that key.

The gate $\alpha_t$ multiplies the whole memory, like Mamba-2's decay, so the model can wipe stale context quickly (at a topic change) as well as edit single associations.

## Complexity
$O(T H d_k d_v)$ time, $O(H d_k d_v)$ memory. Real implementations train with a chunked parallel form: the product of $(I - \beta \hat k \hat k^\top)$ factors within a chunk is rewritten in WY (Householder-like) form so most work becomes matmuls.

## Edge cases
- Zero keys or queries (test 6): the $\max(\lVert\cdot\rVert, 10^{-6})$ guard keeps them at zero instead of NaN, so nothing is written or read.
- $\beta = 0$ (test 8): the memory only decays; the error still reports what it would have corrected.

## Common bugs
- Not normalising $k$: with $\lVert k \rVert > \sqrt{2/\beta}$ the update $I - \beta k k^\top$ has an eigenvalue below $-1$ and the state explodes.
- Reading `old` from the un-decayed state, or computing the output before the write.

## Interview follow-ups
- Why does the delta rule improve associative recall over linear attention with the same state size?
- What are the eigenvalues of $I - \beta \hat k \hat k^\top$, and why do they guarantee the state stays bounded when $\alpha \le 1$?
"""
