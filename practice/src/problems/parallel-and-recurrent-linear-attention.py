import numpy as np

META = dict(title='Causal linear attention: parallel and recurrent forms', tags=['Attention', 'Language Models'],
            diff='Medium', runtime='numpy', lessons=['aieng:07.02', 'aieng:04.05'],
            related=['scaled-dot-product-attention', 'mamba-2-gated-state-scan', 'gated-deltanet-state-update'])

STATEMENT = r"""
Linear attention replaces $\exp(q \cdot k)$ with a product of feature maps $\phi(q)\cdot\phi(k)$. The causal output can then be computed two ways that give the same numbers: a **parallel** form with a masked $T \times T$ matrix (good for training), and a **recurrent** form with a fixed-size state (good for generation, $O(1)$ memory per token).

Use $\phi(x) = \mathrm{elu}(x) + 1$ elementwise, i.e. $\phi(x) = x + 1$ for $x > 0$ and $e^{x}$ otherwise. It is always positive.

Write `linear_attention(Q, K, V, state=None)`.

## Inputs
- `Q`, `K`: `float64` `(T, d)`; `V`: `float64` `(T, d_v)`; $T \ge 1$.
- `state`: `None` (start empty) or a tuple `(S0, z0)` with `S0` `(d, d_v)` and `z0` `(d,)`: the state left by an earlier chunk of the same sequence.

## Definition
With $S_0, z_0$ the initial state (zeros if `state` is `None`):
$$S_t = S_{t-1} + \phi(k_t)\, v_t^\top, \qquad z_t = z_{t-1} + \phi(k_t), \qquad o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}$$
Token $t$ sees itself and everything before it (including the previous chunks summarised by the state).

## Output
A tuple `(O_par, O_rec, S_T, z_T)`:
- `O_par` `(T, d_v)`: computed with matrices and **no loop over time**: a lower-triangular mask of $\phi(Q)\phi(K)^\top$, plus the contribution of the initial state.
- `O_rec` `(T, d_v)`: computed with the recurrence above, one token at a time.
- `S_T`, `z_T`: the final state, to pass to the next chunk.
"""

STARTER = '''import numpy as np

def linear_attention(Q, K, V, state=None):
    """Return (O_par, O_rec, S_T, z_T)."""
    A = Q @ K.T                                  # TODO: no feature map, no causal mask, no state
    O = A @ V / A.sum(axis=1, keepdims=True)
    return O, O, K.T @ V, K.sum(axis=0)
'''

SOLUTION = '''import numpy as np

def _phi(x):
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))   # elu + 1, no overflow warning for big x

def linear_attention(Q, K, V, state=None):
    T, d = Q.shape
    S0, z0 = state if state is not None else (np.zeros((d, V.shape[1])), np.zeros(d))
    fq, fk = _phi(Q), _phi(K)

    # parallel: causal mask on the (T, T) similarity matrix, plus what the earlier chunks left behind
    A = np.tril(fq @ fk.T)
    O_par = (fq @ S0 + A @ V) / (fq @ z0 + A.sum(axis=1))[:, None]

    # recurrent: constant-size state, one token at a time
    S, z = S0.copy(), z0.copy()
    O_rec = np.empty((T, V.shape[1]))
    for t in range(T):
        S += np.outer(fk[t], V[t])
        z += fk[t]
        O_rec[t] = fq[t] @ S / (fq[t] @ z)
    return O_par, O_rec, S, z
'''

HINTS = [
    '$\\phi(q_t)^\\top S_t = \\sum_{s \\le t} (\\phi(q_t)\\cdot\\phi(k_s))\\, v_s$. That sum is row $t$ of `tril(phi(Q) @ phi(K).T) @ V`.',
    'The denominator is the same sum with $v_s$ replaced by 1: the row sums of the masked matrix. The initial state adds `phi(Q) @ S0` to the numerator and `phi(Q) @ z0` to the denominator.',
    'For the recurrence, update `S += outer(phi(k_t), v_t)` and `z += phi(k_t)` **before** computing $o_t$, so token $t$ sees itself. Copy the initial state so the caller\'s arrays are not modified.',
]

_r = np.random.default_rng(4)
_Q, _K, _V = _r.normal(size=(8, 4)), _r.normal(size=(8, 4)), _r.normal(size=(8, 3))
_fk = np.where(_K > 0, _K + 1, np.exp(np.minimum(_K, 0)))
_S4, _z4 = _fk[:4].T @ _V[:4], _fk[:4].sum(axis=0)       # state after the first 4 tokens of test 3
TESTS = [
    (np.array([[0.0], [0.0]]), np.array([[0.0], [0.0]]), np.array([[1.0], [3.0]])),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, -1.0], [0.0, 2.0], [-1.0, 0.0]]),
     np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])),
    (_Q, _K, _V),
    (_Q[4:], _K[4:], _V[4:], (_S4, _z4)),          # continues test 3: same outputs as its last 4 rows
    (_r.normal(size=(1, 5)), _r.normal(size=(1, 5)), _r.normal(size=(1, 2))),
    (_r.normal(size=(20, 6)) * 3, _r.normal(size=(20, 6)) * 3, _r.normal(size=(20, 5))),
    (np.full((3, 2), 50.0), np.full((3, 2), -50.0), np.array([[1.0], [2.0], [4.0]])),
    (_r.normal(size=(6, 3)), _r.normal(size=(6, 3)), _r.normal(size=(6, 4)), (np.ones((3, 4)), np.full(3, 2.0))),
]

EXPLANATION = r"""
## Why the two forms agree
Softmax attention needs every $\exp(q_t \cdot k_s)$ separately. With a feature map, $\phi(q_t)\cdot\phi(k_s)$ factorises, so the sum over $s$ can be moved inside: $\sum_s (\phi(q_t)\cdot\phi(k_s)) v_s = \phi(q_t)^\top \sum_s \phi(k_s) v_s^\top = \phi(q_t)^\top S_t$. The parallel form evaluates the left side for all $t$ at once with a triangular mask; the recurrent form keeps the running sum $S_t$.

Test 4 continues test 3 from its state after 4 tokens and reproduces test 3's last four output rows: chunked processing is exact.

## Complexity
- Parallel: $O(T^2 d)$ time and $O(T^2)$ memory, like softmax attention, but with large matmuls that use tensor cores.
- Recurrent: $O(T d\, d_v)$ time and $O(d\, d_v)$ memory, independent of context length.
- Real kernels use the **chunked** form: parallel inside chunks of 64 to 256 tokens, recurrent across chunks, which is what the `state` argument enables.

## Common bugs
- Forgetting the positivity of $\phi$: with a plain identity map the denominator can be zero or negative.
- Updating the state after computing $o_t$, which makes token $t$ blind to itself (an exclusive scan instead of inclusive).
- Using `np.exp(x)` for all $x$ inside `np.where`: both branches are evaluated, so $e^{50}$ is computed (harmless here) and $e^{1000}$ overflows with a warning (test 7 uses $\pm 50$ to keep it visible but finite).

## Interview follow-ups
- Why does linear attention recall worse than softmax attention on associative-recall tasks? (The state is a fixed $d \times d_v$ matrix: at most $d$ key/value pairs can be stored without interference.)
- How do decay gates (RetNet, Mamba-2, GLA) change the recurrence, and how does the parallel form change with them? (The mask becomes a matrix of cumulative decay products instead of ones.)
"""
