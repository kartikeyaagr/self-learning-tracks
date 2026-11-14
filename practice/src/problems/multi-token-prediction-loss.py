import numpy as np

META = dict(title='Multi-token prediction loss with sequential modules', tags=['Language Models', 'Loss Functions'], diff='Hard',
            runtime='numpy', lessons=['ieng:16.06'],
            related=['cross-entropy-logits', 'perplexity-computation', 'aux-loss-free-moe-routing'])

STATEMENT = r"""
Multi-token prediction (MTP) adds small modules that predict tokens further ahead. In the DeepSeek-V3 design the modules are **sequential**: module $k$ takes the hidden state from depth $k-1$ at position $i$, combines it with the embedding of token $t_{i+k}$, and predicts $t_{i+k+1}$. The output head and embedding are shared with the main model.

Write `mtp_losses(h0, tokens, p, lam)` returning `(main_loss, depth_losses, total)`.

## Inputs
- `h0`: `(T, D)` main-model hidden states (before the final norm). `tokens`: `(T,)` the input ids $t_0..t_{T-1}$.
- `p['emb']` `(V, D)`, `p['out_norm']` `(D,)`, `p['W_out']` `(D, V)`: shared embedding and head. `p['mtp']`: list of $K$ module dicts with `norm_h`, `norm_e` `(D,)`, `M` `(2D, D)`, `Wb` `(D, D)`.

## Definitions
- $\mathrm{RMS}_w(x) = w\odot x/\sqrt{\mathrm{mean}(x^2) + 10^{-6}}$. Head: $\mathrm{logits}(h) = \mathrm{RMS}_{out\_norm}(h)\,W_{out}$.
- **Main loss**: mean cross-entropy of $\mathrm{logits}(h^0_i)$ against $t_{i+1}$ for $i = 0..T-2$.
- **Module $k$** ($k = 1..K$), for positions $i = 0..T-k-1$:
  $u_i = [\mathrm{RMS}_{norm\_h}(h^{k-1}_i);\ \mathrm{RMS}_{norm\_e}(\text{emb}[t_{i+k}])]\,M$, then $h^k_i = u_i + \tanh(u_i W_b)$ (a stand-in for the module's Transformer block).
  Its loss is the mean cross-entropy of $\mathrm{logits}(h^k_i)$ against $t_{i+k+1}$ over $i = 0..T-k-2$. If there are no such positions, that depth's loss is 0.0 and the module is not evaluated further.
- `depth_losses` is the list of the $K$ losses; `total` $=$ `main_loss` $+ \frac{\lambda}{K}\sum_k$ `depth_losses[k]`.
"""

STARTER = '''import numpy as np

def mtp_losses(h0, tokens, p, lam):
    """Return (main_loss, depth_losses, total)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _rms(x, w, eps=1e-6):
    return w * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)

def _ce(h, targets, p):
    z = _rms(h, p['out_norm']) @ p['W_out']
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return float(-logp[np.arange(len(targets)), targets].mean())

def mtp_losses(h0, tokens, p, lam):
    h, t = np.asarray(h0, dtype=float), np.asarray(tokens)
    T = len(t)
    main = _ce(h[:T - 1], t[1:], p)
    losses = []
    for k, m in enumerate(p['mtp'], start=1):
        n = T - k - 1                                   # positions with a target t[i+k+1]
        if n <= 0:
            losses.append(0.0)
            continue
        emb = p['emb'][t[k:T]]                          # t[i+k] for i = 0..T-k-1
        u = np.concatenate([_rms(h[:T - k], m['norm_h']), _rms(emb, m['norm_e'])], axis=-1) @ m['M']
        h = u + np.tanh(u @ m['Wb'])                    # h^k for i = 0..T-k-1
        losses.append(_ce(h[:n], t[k + 1:], p))
    total = main + lam / len(p['mtp']) * sum(losses)
    return main, losses, total
'''

HINTS = [
    'Keep a running array `h` that is one row shorter after each depth: depth $k$ exists for $i = 0..T-k-1$ because it needs $t_{i+k}$.',
    'The embedding input to module $k$ at row $i$ is `emb[tokens[i + k]]`, i.e. `emb[tokens[k:T]]` for all rows at once. The targets are `tokens[k + 1:]`.',
    'Every head call uses the same `out_norm` and `W_out`; only the combine matrix and block differ per module.',
]

_r = np.random.default_rng(1919)
def _p(V, D, K):
    return {'emb': _r.normal(size=(V, D)), 'out_norm': 1 + 0.1 * _r.normal(size=D), 'W_out': _r.normal(size=(D, V)) / np.sqrt(D),
            'mtp': [dict(norm_h=1 + 0.1 * _r.normal(size=D), norm_e=1 + 0.1 * _r.normal(size=D),
                         M=_r.normal(size=(2 * D, D)) / np.sqrt(2 * D), Wb=_r.normal(size=(D, D)) / np.sqrt(D)) for _ in range(K)]}
_p1, _p2 = _p(7, 4, 1), _p(11, 6, 3)
TESTS = [
    (_r.normal(size=(5, 4)), np.array([1, 3, 2, 6, 0]), _p1, 0.3),
    (_r.normal(size=(3, 4)), np.array([4, 4, 4]), _p1, 1.0),
    (_r.normal(size=(8, 6)), np.array([1, 5, 9, 2, 2, 7, 10, 3]), _p2, 0.3),
    (_r.normal(size=(8, 6)), np.array([1, 5, 9, 2, 2, 7, 10, 3]), _p2, 0.0),
    (_r.normal(size=(4, 6)), np.array([3, 8, 1, 6]), _p2, 0.5),
    (_r.normal(size=(2, 4)), np.array([2, 5]), _p1, 0.3),
]

EXPLANATION = r"""
## Why sequential modules
Parallel MTP heads (one head per offset, all reading the same hidden state) ask one representation to encode several future tokens independently. Chaining the modules keeps a causal chain: module $k$ sees the true token $t_{i+k}$ (teacher forcing) and the previous depth's state, so each prediction is conditioned on everything before it, as in ordinary next-token prediction.

## How it is used
- **Training signal**: the extra losses densify supervision and encourage representations that plan ahead. They are weighted by a small $\lambda$ (DeepSeek-V3 used 0.3 early in training and 0.1 later).
- **Inference**: the modules can be dropped, leaving the main model unchanged, or reused as a draft model for speculative decoding.

## Index bookkeeping
For $T$ tokens the main model has $T-1$ targets and depth $k$ has $T-k-1$. Getting the off-by-one wrong silently trains the module to predict the token it was just given.

## Complexity
Each depth adds one block and one vocabulary projection over $\approx T$ positions; with a large vocabulary the projection dominates.

## Common bugs
- Feeding $t_{i+k+1}$ (the target) as the embedding input.
- Computing every depth from $h^0$ rather than chaining $h^{k-1}$.
- Averaging the MTP losses over $K$ but forgetting $\lambda$, or vice versa.

## Interview follow-ups
- How would you turn the trained MTP module into a speculative-decoding draft? What acceptance rate would you expect?
- Why share the embedding and output head instead of giving each module its own?
"""
