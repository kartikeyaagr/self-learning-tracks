import numpy as np

META = dict(title='Capstone: a hybrid stack of sliding-window and gated delta-rule layers', tags=['Attention', 'Transformers'], diff='Hard', runtime='numpy',
            related=['sliding-window-causal-mask', 'gated-deltanet-state-update', 'parallel-and-recurrent-linear-attention', 'hybrid-cache-memory'])

STATEMENT = r"""
Write `hybrid_forward(x, layers, window)` for inputs `x` of shape `(T, d)`. Each layer is a dict with a `"type"` and weights; every layer is applied as $x \leftarrow x + f(\mathrm{RMSNorm}(x))$ with $\mathrm{RMSNorm}(u) = u / \sqrt{\mathrm{mean}(u^2) + 10^{-6}}$ (no learned scale).

- `"S"` (single-head sliding-window attention): $q, k, v = uW_q, uW_k, uW_v$ (each `(d, d)`), scores $qk^\top/\sqrt d$, visible keys $j \le t$ with $t - j <$ `window`, softmax, then $\cdot W_o$.
- `"D"` (gated delta rule, one head): $q_t, k_t$ from $uW_q, uW_k$ are L2-normalised; $v_t = u_tW_v$; gates $\alpha_t = \sigma(u_t \cdot w_\alpha)$, $\beta_t = \sigma(u_t \cdot w_\beta)$ with vectors $w_\alpha, w_\beta$ of length $d$. With state $S_0 = 0$ of shape `(d, d)`:
$$S_t = \alpha_t S_{t-1} (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top, \qquad o_t = S_t q_t$$
  then output $o W_o$.

Return `(y, states)` where `states` is a list with the final $S_T$ of every `"D"` layer in order (the only thing such a layer needs to cache).
"""

STARTER = '''import numpy as np

def hybrid_forward(x, layers, window):
    """Run a stack of sliding-window attention and gated delta-rule layers."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _rms(u):
    return u / np.sqrt((u ** 2).mean(-1, keepdims=True) + 1e-6)

def _swa(u, L, window):
    T, d = u.shape
    q, k, v = u @ L["Wq"], u @ L["Wk"], u @ L["Wv"]
    s = q @ k.T / np.sqrt(d)
    t, j = np.arange(T)[:, None], np.arange(T)[None, :]
    s = np.where((j <= t) & (t - j < window), s, -np.inf)
    a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
    return (a @ v) @ L["Wo"]

def _delta(u, L):
    T, d = u.shape
    nq = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
    q, k, v = nq(u @ L["Wq"]), nq(u @ L["Wk"]), u @ L["Wv"]
    sig = lambda z: 1 / (1 + np.exp(-z))
    al, be = sig(u @ L["wa"]), sig(u @ L["wb"])
    S, out, I = np.zeros((d, d)), np.zeros((T, d)), np.eye(d)
    for t in range(T):
        S = al[t] * S @ (I - be[t] * np.outer(k[t], k[t])) + be[t] * np.outer(v[t], k[t])
        out[t] = S @ q[t]
    return out @ L["Wo"], S

def hybrid_forward(x, layers, window):
    x = np.asarray(x, float).copy()
    states = []
    for L in layers:
        L = {k: (np.asarray(v, float) if k != "type" else v) for k, v in L.items()}
        u = _rms(x)
        if L["type"] == "S":
            x = x + _swa(u, L, window)
        else:
            o, S = _delta(u, L)
            x = x + o
            states.append(S)
    return x, states
'''

HINTS = [
    'Write each layer type as its own function of the normalised input, then loop over the stack with the residual.',
    'The delta rule first erases what the state stores along k (the I - beta k k^T factor), then writes v along k.',
    'Only the delta-rule state has a fixed size; the sliding-window layer would cache its last `window` keys and values.',
]

def _layers(seed, d, pattern):
    r = np.random.default_rng(seed)
    out = []
    for c in pattern:
        L = {"type": c, "Wq": r.normal(size=(d, d)) / np.sqrt(d), "Wk": r.normal(size=(d, d)) / np.sqrt(d),
             "Wv": r.normal(size=(d, d)) / np.sqrt(d), "Wo": r.normal(size=(d, d)) / np.sqrt(d)}
        if c == "D":
            L["wa"], L["wb"] = r.normal(size=d), r.normal(size=d)
        out.append(L)
    return out

_x = lambda seed, T, d: np.random.default_rng(seed + 100).normal(size=(T, d))
TESTS = [
    (_x(1, 5, 4), _layers(1, 4, "S"), 2),
    (_x(2, 6, 4), _layers(2, 4, "D"), 3),
    (_x(3, 8, 6), _layers(3, 6, "DDS"), 3),
    (_x(4, 8, 6), _layers(4, 6, "SDSD"), 8),
    (_x(5, 4, 3), _layers(5, 3, "DS"), 1),
]

EXPLANATION = r"""
## Two ways to remember
Sliding-window attention remembers the last $W$ tokens exactly. The delta rule compresses the whole history into a $d \times d$ matrix, overwriting what it stored along each new key: $S(I - \beta kk^\top)$ removes the old value for $k$, and $\beta vk^\top$ writes the new one. The gate $\alpha$ lets the layer forget everything at once when the context changes.

## Why stack them
The recurrent layers give unbounded but lossy memory at constant cost; the attention layers give precise local recall. Hybrid models use mostly recurrent or windowed layers and a few full-attention layers to recover exact long-range retrieval.

## Testing it
The recurrent form here is $O(Td^2)$ and sequential. Training kernels use a chunked form that is mathematically equal; checking a chunked implementation against this loop on random inputs is the standard test.

## Interview follow-ups
- What does $\beta = 1$ with orthogonal keys reduce the delta rule to?
"""
