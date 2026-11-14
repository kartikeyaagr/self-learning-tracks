import numpy as np

META = dict(title='Capstone: a full mixture-of-experts layer with balancing', tags=['Language Models', 'Transformers'], diff='Hard', runtime='numpy',
            related=['sparse-moe-top-k-expert-routing', 'moe-token-dispatch-and-expert-aggregation', 'switch-ffn-layer-top1', 'aux-loss-free-moe-routing'])

STATEMENT = r"""
Write `moe_layer(x, Wr, bias, experts, k, cap_factor, bias_lr)` for tokens `x` of shape `(T, d)`.

- **Router.** Logits $z = x W_r$ (shape `(T, E)`), probabilities $p = \mathrm{softmax}(z)$. Expert **selection** uses $z + $ `bias` (top-`k` per token, ties to the lower expert id); the combine **weights** are the selected $p$ values renormalised to sum to 1. The bias never affects the weights.
- **Capacity.** Each expert takes at most $C = \lceil \text{cap\_factor} \cdot T k / E \rceil$ assignments. Assign in token order, and within a token in rank order (best expert first); an assignment over capacity is dropped.
- **Experts.** Expert $e$ is a ReLU MLP: $\mathrm{relu}(x W_1^{(e)} + b_1^{(e)}) W_2^{(e)} + b_2^{(e)}$, from `experts[e] = (W1, b1, W2, b2)`.
- **Output.** $y_t = x_t + \sum$ over kept assignments of weight $\times$ expert output (the residual carries dropped tokens through).
- **Losses.** Load fraction $f_e$ = share of all $Tk$ selections (before capacity) that went to $e$; $P_e$ = mean router probability of $e$. Balance loss $= E \sum_e f_e P_e$. Router z-loss $= \mathrm{mean}_t(\log\sum_e e^{z_{te}})^2$.
- **Bias update (auxiliary-loss-free balancing).** $\text{bias}_e \mathrel{+}= \text{bias\_lr} \cdot \mathrm{sign}(\bar c - c_e)$ where $c_e$ is the number of selections of $e$ and $\bar c$ their mean.

Return `(y, dropped, balance_loss, z_loss, new_bias)` where `dropped` is the number of dropped assignments.
"""

STARTER = '''import numpy as np

def moe_layer(x, Wr, bias, experts, k, cap_factor, bias_lr):
    """Return (y, dropped, balance_loss, z_loss, new_bias)."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

def moe_layer(x, Wr, bias, experts, k, cap_factor, bias_lr):
    x, Wr, bias = np.asarray(x, float), np.asarray(Wr, float), np.asarray(bias, float)
    T, E = len(x), Wr.shape[1]
    z = x @ Wr
    m = z.max(1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]
    p = np.exp(z - lse[:, None])
    sel = z + bias
    choice = np.stack([np.lexsort((np.arange(E), -sel[t]))[:k] for t in range(T)])
    w = np.take_along_axis(p, choice, 1)
    w = w / w.sum(1, keepdims=True)
    C = math.ceil(cap_factor * T * k / E)
    load, y, dropped = np.zeros(E, int), x.copy(), 0
    for t in range(T):
        for r in range(k):
            e = choice[t, r]
            if load[e] >= C:
                dropped += 1
                continue
            load[e] += 1
            W1, b1, W2, b2 = (np.asarray(a, float) for a in experts[e])
            y[t] += w[t, r] * (np.maximum(x[t] @ W1 + b1, 0) @ W2 + b2)
    counts = np.bincount(choice.ravel(), minlength=E)
    f = counts / (T * k)
    balance = float(E * (f * p.mean(0)).sum())
    zloss = float((lse ** 2).mean())
    new_bias = bias + bias_lr * np.sign(counts.mean() - counts)
    return y, dropped, balance, zloss, new_bias
'''

HINTS = [
    'Keep two quantities apart: the biased scores choose experts, the plain softmax weights combine them.',
    'Count load for capacity as you go; count selections for the losses and the bias from all choices, kept or not.',
    'A uniform router gives balance loss 1.0, the minimum for a perfectly balanced load.',
]

def _case(seed, T, d, E, h):
    r = np.random.default_rng(seed)
    ex = [(r.normal(size=(d, h)) * 0.5, r.normal(size=h) * 0.1, r.normal(size=(h, d)) * 0.5, r.normal(size=d) * 0.1) for _ in range(E)]
    return r.normal(size=(T, d)), r.normal(size=(d, E)), np.zeros(E), ex

TESTS = [
    _case(1, 6, 4, 4, 8) + (2, 1.0, 0.01),
    _case(2, 10, 4, 4, 8) + (1, 1.0, 0.01),
    _case(3, 10, 4, 4, 8) + (2, 0.5, 0.1),
    (_case(4, 8, 3, 3, 4)[0], _case(4, 8, 3, 3, 4)[1], np.array([0.0, 5.0, 0.0]), _case(4, 8, 3, 3, 4)[3], 1, 2.0, 0.05),
    _case(5, 16, 8, 8, 16) + (2, 1.25, 0.001),
]

EXPLANATION = r"""
## One layer, three balancing tools
- Capacity limits stop any expert from receiving more than its share of a batch; overflow tokens skip the layer through the residual.
- The balance loss $E\sum f_e P_e$ pushes the router toward uniform load through gradients, at some cost to the main objective.
- The bias update adjusts selection without gradients: overloaded experts become slightly less likely to be picked next step. Because the bias does not enter the combine weights, it changes which experts run, not how their outputs are mixed.

The z-loss keeps router logits small, which avoids numerical problems in low precision.

## Reading the tests
The fourth case starts with a large bias on expert 1, so it takes every token, capacity drops the overflow, and the bias update starts pulling it back. Dropped assignments are a direct measure of imbalance.

## Training it
In a language model this layer replaces the MLP in some or all blocks. The project's capstone in PyTorch trains such a model and plots per-expert load over time; without balancing, a few experts usually take most tokens within the first few hundred steps.

## Interview follow-ups
- Why does dropping tokens hurt more at inference than in training?
"""
