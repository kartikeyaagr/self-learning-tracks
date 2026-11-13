import numpy as np

META = dict(title='Mamba-2 style selective state scan', tags=['Language Models', 'Attention'], diff='Medium', runtime='numpy',
            lessons=['aieng:07.02', 'air-fanout-v1:08.04'],
            related=['parallel-and-recurrent-linear-attention', 'gated-deltanet-state-update', 'exponential-moving-average-with-scan'])

STATEMENT = r"""
A Mamba-2 layer keeps, for every head, a matrix state $h \in \mathbb R^{P \times N}$ that decays by a **scalar** gate each step and absorbs the current input. The gate depends on the input through a learned step size $\Delta_t$, which is what makes the model "selective": a large $\Delta_t$ forgets the past and writes the present strongly, a small one keeps the state.

Write `ssd_scan(x, dt_raw, dt_bias, A, B, C, D, h0=None)`.

## Inputs (all `float64`)
- `x`: `(T, H, P)`: $H$ heads of width $P$.
- `dt_raw`: `(T, H)`, `dt_bias`: `(H,)`. The step size is $\Delta_{t,h} = \operatorname{softplus}(\text{dt\_raw}_{t,h} + \text{dt\_bias}_h)$, with $\operatorname{softplus}(u) = \log(1 + e^{u})$. Compute it without overflow for $|u|$ up to 1000.
- `A`: `(H,)`, strictly negative. `D`: `(H,)`, a skip connection.
- `B`, `C`: `(T, N)`: input and output projections, shared by all heads.
- `h0`: `None` (zeros) or `(H, P, N)`, the state before the first token.

## Recurrence, for each head $h$ and step $t = 1, \dots, T$
$$a_t = e^{\Delta_t A}, \qquad h_t = a_t\, h_{t-1} + \Delta_t\, x_t B_t^\top, \qquad y_t = h_t\, C_t + D\, x_t$$
where $x_t B_t^\top$ is the outer product `(P, N)` and $h_t C_t$ is a matrix-vector product giving `(P,)`.

## Output
A tuple `(y, h_T)`: `y` of shape `(T, H, P)` and the final state `(H, P, N)`. Do not modify the inputs.
"""

STARTER = '''import numpy as np

def ssd_scan(x, dt_raw, dt_bias, A, B, C, D, h0=None):
    """Return (y (T, H, P), final state (H, P, N))."""
    T, H, P = x.shape
    N = B.shape[1]
    h = np.zeros((H, P, N)) if h0 is None else h0
    y = np.zeros((T, H, P))
    for t in range(T):
        h = h + x[t][:, :, None] * B[t]          # TODO: no decay, no step size, no skip
        y[t] = h @ C[t]
    return y, h
'''

SOLUTION = '''import numpy as np

def _softplus(u):
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))   # log(1 + e^u), stable both ways

def ssd_scan(x, dt_raw, dt_bias, A, B, C, D, h0=None):
    T, H, P = x.shape
    N = B.shape[1]
    dt = _softplus(dt_raw + dt_bias)                  # (T, H) step sizes, all > 0
    decay = np.exp(dt * A)                            # (T, H) in (0, 1) because A < 0
    h = np.zeros((H, P, N)) if h0 is None else h0.copy()
    y = np.empty((T, H, P))
    for t in range(T):
        # (H, 1, 1) * (H, P, N) + (H, 1, 1) * (H, P, 1) * (1, 1, N)
        h = decay[t][:, None, None] * h + (dt[t][:, None] * x[t])[:, :, None] * B[t][None, None, :]
        y[t] = h @ C[t] + D[:, None] * x[t]
    return y, h
'''

HINTS = [
    'Stable softplus: $\\max(u, 0) + \\log(1 + e^{-|u|})$, with `np.log1p`. Then the decay is `np.exp(dt * A)`, a number in $(0, 1)$ per token and head.',
    'Broadcast over heads at once: the decay is `(H, 1, 1)`, the write is `(dt * x)[:, :, None] * B[t][None, None, :]`, shape `(H, P, N)`.',
    'The output reads the **updated** state: `y[t] = h @ C[t] + D[:, None] * x[t]`. Copy `h0` before updating it.',
]

_r = np.random.default_rng(31)


def _case(T, H, P, N, h0=False):
    return (_r.normal(size=(T, H, P)), _r.normal(size=(T, H)), _r.normal(size=H), -np.exp(_r.normal(size=H)),
            _r.normal(size=(T, N)), _r.normal(size=(T, N)), _r.normal(size=H),
            *((_r.normal(size=(H, P, N)),) if h0 else ()))


TESTS = [
    (np.array([[[1.0]], [[2.0]]]), np.zeros((2, 1)), np.zeros(1), np.array([-1.0]),
     np.ones((2, 1)), np.ones((2, 1)), np.zeros(1)),
    (np.array([[[1.0, -1.0]]]), np.array([[0.5]]), np.array([0.0]), np.array([-1.0]),
     np.array([[1.0, 2.0]]), np.array([[0.5, 0.5]]), np.array([2.0])),
    _case(6, 2, 3, 4),
    _case(5, 3, 2, 2, True),
    (np.ones((3, 1, 1)), np.array([[1000.0], [-1000.0], [0.0]]), np.zeros(1), np.array([-1.0]),
     np.ones((3, 1)), np.ones((3, 1)), np.zeros(1)),
    _case(1, 4, 2, 3, True),
    _case(16, 2, 4, 8),
]

EXPLANATION = r"""
## Why it works
The state is a decaying sum of rank-one writes: unrolling the recurrence, $h_t = \sum_{s \le t} \bigl(\prod_{r=s+1}^{t} a_r\bigr) \Delta_s x_s B_s^\top$. Reading it with $C_t$ gives
$$y_t = \sum_{s \le t} \underbrace{(C_t \cdot B_s)\, \textstyle\prod_{r=s+1}^{t} a_r}_{\text{attention-like weight}}\, \Delta_s x_s + D x_t,$$
which is causal linear attention with $C$ as queries, $B$ as keys, $\Delta x$ as values and a decay mask. This "state space duality" is why Mamba-2 can train with matmuls (the quadratic masked form within chunks) and decode with the recurrence.

Because the decay is a scalar per head and step, the mask entries are just $\exp\bigl(A \sum_{r=s+1}^{t} \Delta_r\bigr)$: a cumulative sum, cheap to compute in parallel.

## Complexity
The scan is $O(T H P N)$ time and $O(H P N)$ state memory, independent of the context length. Generation costs the same per token at position 10 or 100 000.

## Common bugs
- `np.log(1 + np.exp(u))` overflows at $u = 1000$ (test 5). The step size should come out as 1000, which makes $a = e^{-1000} = 0$: the state is fully reset to the current token.
- Applying the decay after the write, so the current token is decayed once too often.
- Treating $A$ as a matrix: in Mamba-2 it is a scalar per head (Mamba-1 had a diagonal $A$ per channel).

## Interview follow-ups
- Derive the parallel (chunked) form: within a chunk, how do you compute the decay matrix $L_{ts}$ from a cumulative sum of $\Delta A$, and how does the inter-chunk state get passed along?
- Why is a scalar decay per head, rather than a full diagonal, what makes the matmul form efficient on tensor cores?
"""
