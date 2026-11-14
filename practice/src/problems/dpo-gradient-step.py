import numpy as np

META = dict(title='DPO on a bigram language model: sequence log-probs, loss and one gradient step',
            tags=['Reinforcement Learning', 'Language Models'], diff='Hard', runtime='numpy', lessons=['dlcore:08.07'],
            related=['dpo-closed-form-loss', 'bradley-terry-reward-model-loss', 'compare-sequence-and-token-length-normalization'])

STATEMENT = r"""
Direct Preference Optimisation trains the policy straight from preference pairs, with no reward model and no sampling. Implement one full step for a tiny bigram language model, including the gradient through the sequence log-probabilities.

Write `dpo_step(W, W_ref, pairs, beta, lr)` returning `(loss, W_new, chosen_rewards, rejected_rewards)`.

## Model
$p(y_t\,|\,y_{t-1}) = \mathrm{softmax}(W[y_{t-1}])[y_t]$ with `W` `(V, V)`. A sequence is a list of token ids whose first token is the prompt; its log-probability is $\log\pi(y) = \sum_{t\ge1}\log p(y_t|y_{t-1})$ over the response tokens.

## Loss
`pairs` is a list of `(chosen, rejected)` sequences. With implicit rewards $\hat r(y) = \beta\,(\log\pi_W(y) - \log\pi_{W_{ref}}(y))$:
$$\mathcal L = -\frac1N\sum_{n}\log\sigma\big(\hat r(y_w^{(n)}) - \hat r(y_l^{(n)})\big)$$

## Step
Return the loss (before the update), $W - \text{lr}\,\partial\mathcal L/\partial W$ (`W_ref` is frozen), and the implicit rewards of the chosen and rejected sequences `(N,)` under the **old** `W`.
"""

STARTER = '''import numpy as np

def dpo_step(W, W_ref, pairs, beta, lr):
    """Return (loss, W_new, chosen_rewards, rejected_rewards)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _logsm(M):
    z = M - M.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))

def _seq(L, y):
    y = np.asarray(y)
    return L[y[:-1], y[1:]].sum()

def dpo_step(W, W_ref, pairs, beta, lr):
    W, Wr = np.asarray(W, dtype=float), np.asarray(W_ref, dtype=float)
    L, Lr = _logsm(W), _logsm(Wr)
    P = np.exp(L)
    rc = np.array([beta * (_seq(L, c) - _seq(Lr, c)) for c, _ in pairs])
    rr = np.array([beta * (_seq(L, r) - _seq(Lr, r)) for _, r in pairs])
    m = rc - rr
    loss = float(np.logaddexp(0, -m).mean())
    coef = -beta * (1 - 1 / (1 + np.exp(-m))) / len(m)
    G = np.zeros_like(W)
    for k, (c, r) in enumerate(pairs):
        for y, sgn in ((c, 1.0), (r, -1.0)):
            y = np.asarray(y)
            for a, b in zip(y[:-1], y[1:]):
                G[a] -= sgn * coef[k] * P[a]
                G[a, b] += sgn * coef[k]
    return loss, W - lr * G, rc, rr
'''

HINTS = [
    'Take a row-wise log-softmax of $W$ once; a sequence\'s log-probability is the sum of entries `L[y[t-1], y[t]]`.',
    'With margin $m = \\hat r_w - \\hat r_l$, $\\partial\\mathcal L/\\partial m = -(1 - \\sigma(m))/N$, and $\\partial m/\\partial\\log\\pi(y_w) = \\beta$, $\\partial m/\\partial\\log\\pi(y_l) = -\\beta$.',
    'Each transition $a\\to b$ contributes $\\partial\\log p(b|a)/\\partial W[a] = \\mathbf 1_b - \\mathrm{softmax}(W[a])$; only row $a$ changes.',
]

_r = np.random.default_rng(2204)
def _seqs(V, n, lo, hi):
    return [[int(t) for t in _r.integers(0, V, int(_r.integers(lo, hi + 1)))] for _ in range(n)]
def _c(V, N, lo, hi, beta, lr, same_ref=False):
    Wr = _r.normal(size=(V, V))
    W = Wr.copy() if same_ref else Wr + 0.3 * _r.normal(size=(V, V))
    c, r = _seqs(V, N, lo, hi), _seqs(V, N, lo, hi)
    for k in range(N):
        r[k][0] = c[k][0]                     # same prompt token
    return W, Wr, [(c[k], r[k]) for k in range(N)], beta, lr
TESTS = [
    _c(3, 1, 3, 3, 0.1, 1.0, same_ref=True),
    _c(4, 2, 2, 5, 0.1, 1.0),
    _c(5, 4, 3, 8, 0.5, 0.5),
    _c(6, 8, 4, 10, 0.05, 2.0),
    _c(4, 3, 2, 6, 1.0, 0.1, same_ref=True),
]

EXPLANATION = r"""
## The derivation in one line
The KL-regularised RLHF objective has the closed-form optimum $\pi^*(y|x) \propto \pi_{ref}(y|x)\,e^{r(x,y)/\beta}$. Solving for $r$ gives $r = \beta\log(\pi^*/\pi_{ref}) + \beta\log Z(x)$, and in the Bradley-Terry likelihood of a preference pair the intractable $\log Z(x)$ cancels. So the reward model can be replaced by the policy's own log-ratio, and preference learning becomes a classification loss on the policy.

## What the gradient does
Each pair's gradient is weighted by $1 - \sigma(m)$, how wrong the implicit reward currently ranks the pair, and pushes up the chosen sequence's log-probability and down the rejected one's, token by token. When the policy equals the reference (tests 1 and 5) every margin is 0 and the loss is $\log 2$.

## Practical notes
- Both log-probabilities are summed over the response only; the prompt is conditioning.
- The policy can lower both chosen and rejected likelihoods while still widening the margin, which is a known failure mode; variants add an SFT term on the chosen responses.
- Longer responses get larger-magnitude log-probabilities, which can bias DPO towards length; some variants normalise by length.

## Common bugs
- Forgetting to subtract the reference log-probabilities.
- Flipping the sign of the rejected term in the gradient.
- Updating the reference model.

## Interview follow-ups
- Why is DPO "offline", and what do online or iterative variants change?
- What does $\beta$ control, and what happens as it goes to 0?
"""
