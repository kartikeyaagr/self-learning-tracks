import numpy as np

META = dict(title='Switch Transformer layer: top-1 routing, expert capacity and the balance loss', tags=['Transformers', 'Language Models'],
            diff='Hard', runtime='numpy', lessons=['aieng:06.02', 'ieng:19.02'],
            related=['top-k-moe-router-with-load-statistics', 'moe-token-dispatch-and-expert-aggregation', 'sparse-moe-top-k-expert-routing'])

STATEMENT = r"""
The Switch Transformer replaces the feed-forward block with $E$ experts and sends each token to exactly **one** of them. Each expert has a fixed capacity, and an auxiliary loss pushes the router towards even load. Implement the layer.

Write `switch_ffn(x, Wr, experts, capacity_factor, alpha)` returning `(y, expert, dropped, aux_loss)`.

## Routing
`x` `(T, D)`, `Wr` `(D, E)`. Router probabilities $p = \mathrm{softmax}(xW_r)$ per token; token $t$ picks $e_t = \arg\max_e p_{te}$.

## Capacity
$C = \lfloor \text{capacity\_factor}\cdot T/E\rfloor$ tokens per expert. Tokens claim slots in order of position $t$; a token whose expert is already full is **dropped**.

## Experts
`experts` is a list of $E$ dicts with `W1` `(D, F)`, `b1`, `W2` `(F, D)`, `b2`: $\mathrm{FFN}(u) = \mathrm{ReLU}(uW_1 + b_1)W_2 + b_2$. A kept token's output is $y_t = p_{t e_t}\,\mathrm{FFN}_{e_t}(x_t)$; a dropped token's is $y_t = 0$ (the residual connection outside the layer carries it through).

## Balance loss
$f_e$ = fraction of tokens whose argmax is $e$ (counted **before** dropping), $P_e = \mathrm{mean}_t\,p_{te}$; $\text{aux} = \alpha\,E\sum_e f_eP_e$.

Return `y` `(T, D)`, `expert` `(T,)` ints, `dropped` `(T,)` as 0/1 ints and `aux_loss` as a float.
"""

STARTER = '''import numpy as np

def switch_ffn(x, Wr, experts, capacity_factor, alpha):
    """Return (y, expert, dropped, aux_loss)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def switch_ffn(x, Wr, experts, capacity_factor, alpha):
    x = np.asarray(x, dtype=float)
    T, E = x.shape[0], len(experts)
    lg = x @ np.asarray(Wr, dtype=float)
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ex = p.argmax(1)
    C = int(np.floor(capacity_factor * T / E))
    used = np.zeros(E, dtype=int)
    dropped = np.zeros(T, dtype=int)
    y = np.zeros_like(x)
    for t in range(T):
        e = ex[t]
        if used[e] >= C:
            dropped[t] = 1
            continue
        used[e] += 1
        P = experts[e]
        h = np.maximum(x[t] @ P['W1'] + P['b1'], 0)
        y[t] = p[t, e] * (h @ P['W2'] + P['b2'])
    f = np.bincount(ex, minlength=E) / T
    aux = float(alpha * E * (f * p.mean(0)).sum())
    return y, ex, dropped, aux
'''

HINTS = [
    'Softmax the router logits, take the argmax per token, and keep a running count of slots used per expert as you walk the tokens in order.',
    'The expert output is scaled by the router probability of the chosen expert. That multiplication is the only path by which the router gets gradient from the task loss.',
    'In the balance loss, $f_e$ is a hard count (no gradient) and $P_e$ is a soft average; with perfectly uniform routing the sum $E\\sum_e f_eP_e$ equals 1.',
]

_r = np.random.default_rng(2101)
def _experts(E, D, F):
    n = lambda *s: _r.normal(size=s)
    return [dict(W1=n(D, F) / np.sqrt(D), b1=0.1 * n(F), W2=n(F, D) / np.sqrt(F), b2=0.1 * n(D)) for _ in range(E)]
def _router(D, E, scale=1.0):
    return scale * _r.normal(size=(D, E))
_x1 = _r.normal(size=(6, 4))
_ex4 = _experts(4, 4, 8)
TESTS = [
    (_x1, _router(4, 4), _ex4, 2.0, 0.01),
    (_x1, _router(4, 4), _ex4, 1.0, 0.01),
    (_r.normal(size=(16, 4)), _router(4, 4, scale=3.0), _ex4, 1.25, 0.01),
    (_r.normal(size=(12, 6)), _router(6, 3), _experts(3, 6, 5), 1.0, 0.1),
    (_r.normal(size=(10, 5)) + 2.0, _router(5, 2), _experts(2, 5, 4), 0.6, 1.0),
    (_r.normal(size=(8, 3)), _router(3, 8), _experts(8, 3, 4), 4.0, 0.01),
]

EXPLANATION = r"""
## Why top-1
Earlier mixture-of-experts layers routed to the top 2 or more experts, arguing a router needs to compare experts to learn. Switch showed top-1 works, halving expert compute and communication, so a model can have many times more parameters at the same FLOPs per token. The paper reports large pre-training speedups over a dense T5 at equal compute.

## Capacity
Tensor shapes must be static on accelerators, so each expert gets a fixed buffer of $C$ slots. Too small a capacity factor drops tokens (they skip the layer through the residual); too large wastes compute and memory on padding. The paper uses factors around 1.0 to 1.25. Tests 1 and 6 drop nothing; test 5 sends every token to one expert, the collapse the balance loss is there to prevent, and drops 7 of 10.

## The balance loss
Without pressure, the router tends to collapse onto a few experts that then get all the training signal. $E\sum_e f_eP_e$ is minimised under uniform routing; $f_e$ isn't differentiable, so gradient flows only through $P_e$, pushing probability mass away from overloaded experts. $\alpha = 0.01$ was the paper's default.

## Common bugs
- Computing $f_e$ after dropping, which hides overload from the loss.
- Forgetting to scale the expert output by the router probability.
- Filling capacity in an order other than token position (implementations differ, but the spec fixes it).

## Interview follow-ups
- How does auxiliary-loss-free balancing (a per-expert bias on the router logits) change this?
- What does expert parallelism require at the all-to-all step, and how does capacity bound it?
"""
