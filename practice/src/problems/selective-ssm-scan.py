import numpy as np

META = dict(title='Selective state-space scan (Mamba S6)', tags=['Language Models', 'Attention'], diff='Medium', runtime='numpy',
            related=['mamba-2-gated-state-scan', 'parallel-and-recurrent-linear-attention', 'gated-deltanet-state-update'])

STATEMENT = r"""
A state-space layer keeps a hidden state per channel and updates it linearly at each step. Mamba makes the step size and the input/output projections functions of the current token (the **selection** mechanism), so the layer can choose what to remember. Implement the scan.

Write `selective_scan(x, delta, A, B, C, D, h0)` returning `(y, h)`.

## Shapes
`x`, `delta` `(L, Dc)`; `A` `(Dc, N)` (negative entries); `B`, `C` `(L, N)`; `D` `(Dc,)`; `h0` `(Dc, N)`.

## Discretise, then scan (for $t = 0..L-1$)
- $\bar A_t = \exp(\delta_t[:, \text{None}]\odot A)$ `(Dc, N)` (zero-order hold).
- $\bar B_t x_t = (\delta_t \odot x_t)[:, \text{None}]\;B_t[\text{None}, :]$ (the simplified first-order input term Mamba uses).
- $h_t = \bar A_t\odot h_{t-1} + \bar B_t x_t$, starting from $h_{-1} =$ `h0`.
- $y_t = h_t C_t + D\odot x_t$, i.e. $y_t[d] = \sum_n h_t[d, n]\,C_t[n] + D[d]\,x_t[d]$.

Return `y` `(L, Dc)` and the final state `h` `(Dc, N)`.
"""

STARTER = '''import numpy as np

def selective_scan(x, delta, A, B, C, D, h0):
    """Return (y (L, Dc), h (Dc, N))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def selective_scan(x, delta, A, B, C, D, h0):
    x, delta, A, B, C, D = (np.asarray(a, dtype=float) for a in (x, delta, A, B, C, D))
    h = np.array(h0, dtype=float)
    y = np.zeros_like(x)
    for t in range(len(x)):
        h = np.exp(delta[t][:, None] * A) * h + (delta[t] * x[t])[:, None] * B[t][None, :]
        y[t] = h @ C[t] + D * x[t]
    return y, h
'''

HINTS = [
    'Each channel $d$ has its own $N$-dimensional state; all channels share $B_t$ and $C_t$ at step $t$, but each has its own step size $\\delta_t[d]$.',
    'With $A < 0$ and $\\delta > 0$, $\\bar A_t$ lies in $(0, 1)$: a large step size forgets the past quickly and lets the current input in strongly, a small one keeps the state.',
    'Return the final state too: passing it as `h0` to the next call continues the sequence exactly, which is how Mamba decodes one token at a time.',
]

_r = np.random.default_rng(2102)
def _c(L, Dc, N, zero_h=True):
    sp = lambda u: np.log1p(np.exp(u))
    return (_r.normal(size=(L, Dc)), sp(_r.normal(size=(L, Dc)) - 1), -np.exp(_r.normal(size=(Dc, N)) * 0.5),
            _r.normal(size=(L, N)), _r.normal(size=(L, N)), _r.normal(size=Dc),
            np.zeros((Dc, N)) if zero_h else _r.normal(size=(Dc, N)))
TESTS = [
    _c(1, 2, 2),
    _c(4, 2, 3),
    _c(8, 4, 4),
    _c(6, 3, 2, zero_h=False),
    _c(20, 5, 8),
    _c(3, 1, 1, zero_h=False),
]

EXPLANATION = r"""
## From S4 to S6
Earlier structured state-space models (S4) used fixed $A, B, C, \delta$, which made the layer a linear time-invariant system: it can be computed as one long convolution, but it treats every token the same and can't decide to ignore filler or to latch onto a key token. Making $\delta, B, C$ depend on the input breaks the convolution view but adds content-based selection.

## Cost
Training uses a hardware-aware parallel scan (the recurrence is associative) that keeps the expanded state in on-chip memory. At inference the state is a fixed $D_c\times N$ array per layer, so each new token costs $O(D_cN)$ regardless of how long the context is, unlike attention's growing KV cache.

## Discretisation
Zero-order hold gives $\bar A = e^{\delta A}$ exactly; the input term's exact form is $(\delta A)^{-1}(e^{\delta A} - 1)\,\delta B$, which Mamba approximates by $\delta B$.

## Common bugs
- Sharing one $\delta$ across channels, or applying it to $B$ but not $A$.
- Forgetting the skip term $D\odot x$.
- Updating $y$ from the previous state instead of the new one.

## Interview follow-ups
- Mamba-2 restricts $A$ to a scalar times identity per head. What does that buy?
- Why do hybrid models interleave a few attention layers with many SSM layers?
"""
