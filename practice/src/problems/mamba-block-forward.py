import numpy as np

META = dict(title='Mamba block forward: projection, causal conv, selective scan and gate', tags=['Language Models', 'Neural Networks'],
            diff='Hard', runtime='numpy',
            related=['selective-ssm-scan', 'mamba-2-gated-state-scan', 'depthwise-separable-convolution', 'parameter-matched-swiglu-block'])

STATEMENT = r"""
A Mamba block wraps the selective scan in a gated architecture: expand, mix locally with a short causal convolution, run the input-dependent SSM, gate, and project back. Implement the forward pass for one sequence.

Write `mamba_block(x, p)` returning `out` `(L, Dm)`.

## Steps
`x` `(L, Dm)`; inner width $D_i$, state size $N$, rank $R$.
1. $[u\,|\,z] = xW_{in}$ with `W_in` `(Dm, 2Di)` ($u$ is the first half).
2. Causal depthwise conv, kernel width $K$: $c_t[d] = b^{conv}_d + \sum_{k=0}^{K-1} W^{conv}_{d,k}\,u_{t-K+1+k}[d]$ with $u_s = 0$ for $s < 0$; then $u \leftarrow \mathrm{SiLU}(c)$. `conv_w` `(Di, K)`, `conv_b` `(Di,)`.
3. $[\delta_{low}\,|\,B\,|\,C] = uW_x$ with `W_x` `(Di, R + 2N)` (widths $R$, $N$, $N$).
4. $\delta = \mathrm{softplus}(\delta_{low}W_{dt} + b_{dt})$, `W_dt` `(R, Di)`; $A = -\exp(A_{log})$, `A_log` `(Di, N)`.
5. Selective scan from a zero state, per channel $d$: $h_t = e^{\delta_t A}\odot h_{t-1} + (\delta_t\odot u_t)B_t^\top$; $y_t = h_tC_t + D\odot u_t$.
6. $\text{out} = (y\odot\mathrm{SiLU}(z))\,W_{out}$, `W_out` `(Di, Dm)`.

`p` holds `W_in conv_w conv_b W_x W_dt b_dt A_log D W_out`. Use a numerically stable softplus.
"""

STARTER = '''import numpy as np

def mamba_block(x, p):
    """Return out (L, Dm)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _silu(u):
    return u / (1 + np.exp(-u))

def mamba_block(x, p):
    x = np.asarray(x, dtype=float)
    L = x.shape[0]
    Di, K = p['conv_w'].shape
    N = p['A_log'].shape[1]
    R = p['W_dt'].shape[0]
    u, z = np.split(x @ p['W_in'], 2, axis=1)
    up = np.vstack([np.zeros((K - 1, Di)), u])
    c = p['conv_b'] + sum(p['conv_w'][:, k] * up[k:k + L] for k in range(K))
    u = _silu(c)
    proj = u @ p['W_x']
    dl, B, C = proj[:, :R], proj[:, R:R + N], proj[:, R + N:]
    pre = dl @ p['W_dt'] + p['b_dt']
    delta = np.logaddexp(0.0, pre)
    A = -np.exp(p['A_log'])
    h = np.zeros((Di, N))
    y = np.zeros((L, Di))
    for t in range(L):
        h = np.exp(delta[t][:, None] * A) * h + (delta[t] * u[t])[:, None] * B[t][None, :]
        y[t] = h @ C[t] + p['D'] * u[t]
    return (y * _silu(z)) @ p['W_out']
'''

HINTS = [
    'Left-pad $u$ with $K - 1$ zero rows; then output row $t$ uses padded rows $t..t+K-1$, so the last kernel tap multiplies the current token.',
    'The scan is the same as a standalone selective scan: $\\delta$ is per channel and per step, $B_t$ and $C_t$ are shared across channels. Softplus as `np.logaddexp(0, x)` is stable.',
    'The gate branch $z$ skips the convolution and the SSM entirely; it only multiplies the SSM output through SiLU.',
]

_r = np.random.default_rng(2103)
def _p(Dm, Di, N, R, K):
    n = lambda *s: _r.normal(size=s)
    return dict(W_in=n(Dm, 2 * Di) / np.sqrt(Dm), conv_w=n(Di, K) * 0.5, conv_b=0.1 * n(Di), W_x=n(Di, R + 2 * N) / np.sqrt(Di),
                W_dt=n(R, Di) / np.sqrt(R), b_dt=n(Di) * 0.5 - 1.0,
                A_log=np.log(np.tile(np.arange(1, N + 1, dtype=float), (Di, 1))), D=np.ones(Di), W_out=n(Di, Dm) / np.sqrt(Di))
_p1, _p2, _p3 = _p(2, 4, 2, 1, 2), _p(4, 8, 4, 2, 4), _p(3, 6, 3, 2, 3)
TESTS = [
    (_r.normal(size=(1, 2)), _p1),
    (_r.normal(size=(3, 2)), _p1),
    (_r.normal(size=(6, 4)), _p2),
    (_r.normal(size=(2, 4)), _p2),
    (_r.normal(size=(10, 3)), _p3),
]

EXPLANATION = r"""
## Architecture
Mamba merges the two halves of earlier SSM architectures (an SSM layer followed by a gated MLP) into one homogeneous block, stacked like transformer layers with a norm and residual around each. The expansion to $D_i \approx 2D_m$ plays the role of the MLP's hidden width. There is no attention and no positional encoding: the convolution and the recurrence carry order.

## Why the short convolution
The scan's input-dependent parameters are computed from $u$ after the convolution, so each token's $\delta, B, C$ see a few neighbouring tokens. That local mixing helps with tasks like copying and induction that a pointwise selection would struggle with.

## Initialisation details
$A_{log}$ is commonly initialised so $A = -[1, 2, \dots, N]$ per channel (as in the tests), giving a range of decay rates; $b_{dt}$ is set so softplus outputs a step size in a modest range.

## Inference
At decode time the block keeps two small caches per layer: the last $K - 1$ conv inputs and the $D_i\times N$ SSM state. Both are constant size, which is why Mamba's memory doesn't grow with context.

## Common bugs
- Non-causal ("same") padding in the convolution, which leaks future tokens.
- Computing $\delta, B, C$ from the pre-convolution input.
- Gating with $\mathrm{SiLU}(y)$ instead of $\mathrm{SiLU}(z)$.

## Interview follow-ups
- What does the decode-time cache contain, and how large is it compared with a KV cache at 32k tokens?
- Which tasks do pure SSMs still do worse on than attention?
"""
