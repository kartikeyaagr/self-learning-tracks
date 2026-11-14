import numpy as np

META = dict(title='A full Gated DeltaNet token-mixing layer', tags=['Language Models', 'Attention'], diff='Hard',
            runtime='numpy', lessons=['aieng:07.02'],
            related=['gated-deltanet-state-update', 'chunkwise-gated-delta-rule', 'mamba-2-gated-state-scan'])

STATEMENT = r"""
The gated delta rule is only the core of a Gated DeltaNet layer. Around it sit input projections, a short causal convolution, normalisation of queries and keys, data-dependent decay and write gates, and a gated output normalisation. Build the whole layer.

Write `gated_deltanet_layer(x, p, n_heads)` returning `(T, D)` for one sequence `x` `(T, D)`.

## Steps
1. **Projections**: $q = xW_q$, $k = xW_k$ `(T, H·d_k)`, $v = xW_v$ `(T, H·d_v)`.
2. **Short convolution + SiLU** on each of $q, k, v$ separately: a causal depthwise convolution with kernel `conv_q`, `conv_k`, `conv_v` of shape `(K, channels)`: $\tilde z_t = \sum_{j=0}^{K-1} w_j \odot z_{t-K+1+j}$ (zero for negative indices), then SiLU.
3. Split into heads (contiguous blocks). Normalise each head's $q$ and $k$ to unit L2 norm (divide by $\max(\lVert\cdot\rVert, 10^{-6})$), then scale $q$ by $1/\sqrt{d_k}$.
4. **Gates** per head: $\beta_t = \sigma(x_tW_\beta)$ `(T, H)`; $\alpha_t = \exp(-e^{A_{log}}\cdot\mathrm{softplus}(x_tW_a + \text{dt\_bias}))$ `(T, H)`, with `A_log`, `dt_bias` of shape `(H,)`.
5. **Recurrence** per head, $S_0 = 0$: $S_t = \alpha_t S_{t-1}(I - \beta_t k_tk_t^\top) + \beta_t v_tk_t^\top$, $o_t = S_tq_t$.
6. **Output**: per head $\mathrm{RMSNorm}(o_t)$ with weight `o_norm` `(d_v,)` and $\epsilon = 10^{-6}$, multiplied by $\mathrm{SiLU}(x_tW_g)$ for that head's slice ($W_g$ `(D, H·d_v)`); concatenate heads and multiply by $W_o$ `(H·d_v, D)`.

Keys: `Wq Wk Wv Wb Wa Wg Wo conv_q conv_k conv_v A_log dt_bias o_norm`. $d_k$ and $d_v$ come from the shapes.
"""

STARTER = '''import numpy as np

def gated_deltanet_layer(x, p, n_heads):
    """Return (T, D)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _silu(z):
    return z / (1 + np.exp(-z))

def _softplus(u):
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))

def _short_conv(z, w):
    K, T = w.shape[0], z.shape[0]
    pad = np.vstack([np.zeros((K - 1, z.shape[1])), z])
    return sum(w[j] * pad[j:j + T] for j in range(K))

def _unit(z):
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-6)

def gated_deltanet_layer(x, p, n_heads):
    x = np.asarray(x, dtype=float)
    T, H = x.shape[0], n_heads
    q = _silu(_short_conv(x @ p['Wq'], p['conv_q']))
    k = _silu(_short_conv(x @ p['Wk'], p['conv_k']))
    v = _silu(_short_conv(x @ p['Wv'], p['conv_v']))
    dk, dv = q.shape[1] // H, v.shape[1] // H
    q = _unit(q.reshape(T, H, dk)) / np.sqrt(dk)
    k = _unit(k.reshape(T, H, dk))
    v = v.reshape(T, H, dv)
    beta = 1 / (1 + np.exp(-(x @ p['Wb'])))
    alpha = np.exp(-np.exp(p['A_log']) * _softplus(x @ p['Wa'] + p['dt_bias']))
    S = np.zeros((H, dv, dk))
    o = np.empty((T, H, dv))
    for t in range(T):
        kt = k[t]
        old = np.einsum('hvk,hk->hv', S, kt)
        S = alpha[t][:, None, None] * (S - beta[t][:, None, None] * old[:, :, None] * kt[:, None, :]) \
            + beta[t][:, None, None] * v[t][:, :, None] * kt[:, None, :]
        o[t] = np.einsum('hvk,hk->hv', S, q[t])
    o = p['o_norm'] * o / np.sqrt((o * o).mean(-1, keepdims=True) + 1e-6)
    o = o * _silu(x @ p['Wg']).reshape(T, H, dv)
    return o.reshape(T, H * dv) @ p['Wo']
'''

HINTS = [
    'Do the short convolution on the full projected width before splitting heads: left-pad with $K-1$ zero rows and sum $K$ shifted copies weighted by the kernel rows.',
    'Expand the recurrence as $S_t = \\alpha_t(S_{t-1} - \\beta_t (S_{t-1}k_t)k_t^\\top) + \\beta_t v_t k_t^\\top$ so each step is two rank-1 updates per head.',
    'The output norm and gate act per head: RMS over the last axis of an `(T, H, d_v)` array, then multiply by the SiLU gate reshaped to the same shape.',
]

_r = np.random.default_rng(2501)
def _model(D, H, dk, dv, K):
    n = lambda *s: _r.normal(size=s)
    return dict(Wq=n(D, H * dk) / np.sqrt(D), Wk=n(D, H * dk) / np.sqrt(D), Wv=n(D, H * dv) / np.sqrt(D),
                Wb=n(D, H) / np.sqrt(D), Wa=n(D, H) / np.sqrt(D), Wg=n(D, H * dv) / np.sqrt(D), Wo=n(H * dv, D) / np.sqrt(H * dv),
                conv_q=n(K, H * dk) * 0.5, conv_k=n(K, H * dk) * 0.5, conv_v=n(K, H * dv) * 0.5,
                A_log=0.3 * n(H), dt_bias=0.3 * n(H), o_norm=1 + 0.1 * n(dv))
_m1, _m2, _m3 = _model(4, 1, 2, 2, 2), _model(6, 2, 3, 2, 4), _model(8, 4, 2, 3, 3)
TESTS = [
    (_r.normal(size=(3, 4)), _m1, 1),
    (_r.normal(size=(6, 4)), _m1, 1),
    (_r.normal(size=(5, 6)), _m2, 2),
    (_r.normal(size=(8, 6)) * 2, _m2, 2),
    (_r.normal(size=(7, 8)), _m3, 4),
    (_r.normal(size=(1, 8)), _m3, 4),
]

EXPLANATION = r"""
## What each piece is for
- **Short convolution**: a cheap way to mix a few neighbouring tokens before the recurrence, which helps recall of local patterns (the recurrent state alone is poor at exact short-range copying).
- **L2-normalised keys** keep $I - \beta k k^\top$ a contraction for $\beta \in [0, 1]$, so the state cannot blow up.
- **$\beta_t$** sets how strongly token $t$ overwrites the value stored under its key; **$\alpha_t$** decays the whole memory, letting the layer clear stale content quickly (the "gated" part, borrowed from Mamba-2's parameterisation).
- **Gated RMSNorm** on the output stabilises scale and lets the layer suppress its own output per channel.

## Where it sits in a model
Gated DeltaNet layers are usually interleaved with a few full or sliding-window attention layers (as in recent hybrid models). The recurrent layers give constant memory per token; the attention layers keep precise long-range retrieval.

## Complexity
Sequential: $O(T H d_k d_v)$ time and $O(H d_k d_v)$ state, independent of context length. Training uses the chunkwise form instead of this loop.

## Common bugs
- Making the convolution non-causal (centred kernel).
- Normalising $q$ and $k$ before the convolution and SiLU instead of after.
- Using $\alpha = \sigma(\cdot)$ or forgetting that $e^{A_{log}}$ multiplies the softplus.

## Interview follow-ups
- Why does a delta-rule memory forget less destructively than plain linear attention's additive state?
- What is the decode-time memory per layer for $H = 16$, $d_k = d_v = 128$, compared with a KV cache at 32k tokens?
"""
