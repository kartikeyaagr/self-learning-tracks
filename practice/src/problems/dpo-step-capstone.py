import numpy as np

META = dict(title='Capstone: one DPO step on a tiny language model', tags=['Language Models', 'Training'], diff='Hard', runtime='numpy',
            related=['score-multiple-choice-continuations', 'dpo-closed-form-loss', 'dpo-reward-accuracy-margin'])

STATEMENT = r"""
Train a tiny language model on preference pairs with DPO, with the gradient written by hand.

The model predicts token $x_{t+1}$ from $x_t$ with logits $E[x_t]\,W$, where `E (V, d)` is frozen and `W (d, V)` is trained. The reference model uses `W_ref` and never changes.

Write `dpo_step(pairs, E, W, W_ref, beta, lr)`. Each pair is `(chosen_ids, rejected_ids, prompt_len)`; both sequences start with the same prompt of length `prompt_len`.

1. **Sequence score.** $\log\pi(y) = \sum_t \log\text{softmax}(E[x_t]W)[x_{t+1}]$ over the positions $t$ with $t+1 \ge$ `prompt_len` (the reply tokens only).
2. **Loss.** $h = (\log\pi_W(y_c) - \log\pi_{ref}(y_c)) - (\log\pi_W(y_r) - \log\pi_{ref}(y_r))$ and $L = \text{mean}(-\log\sigma(\beta h))$.
3. **Gradient.** $\partial \log\pi(y)/\partial W = \sum_t E[x_t]^\top(e_{x_{t+1}} - p_t)$ over the same positions, and $\partial L/\partial h = -\beta\,\sigma(-\beta h)/n$ per pair.
4. **Step.** $W \leftarrow W - \text{lr}\,\nabla_W L$.

Return a dict with `loss_before`, `loss_after`, `margin_before`, `margin_after` (mean of $\beta h$), `accuracy_after` (fraction of pairs with $h > 0$), `chosen_reward_after` and `rejected_reward_after` (means of $\beta(\log\pi_W - \log\pi_{ref})$ after the step).
"""

STARTER = '''import numpy as np

def dpo_step(pairs, E, W, W_ref, beta, lr):
    """One gradient step of DPO on a tiny next-token model."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _score(E, W, ids, pl):
    ids = np.asarray(ids)
    t = np.arange(max(pl - 1, 0), len(ids) - 1)
    z = E[ids[t]] @ W
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    p = np.exp(lp)
    G = -p
    G[np.arange(len(t)), ids[t + 1]] += 1
    return float(lp[np.arange(len(t)), ids[t + 1]].sum()), E[ids[t]].T @ G

def _stats(pairs, E, W, W_ref, beta):
    hs, rc, rr, grads = [], [], [], []
    for c, r, pl in pairs:
        sc, gc = _score(E, W, c, pl); sr, gr = _score(E, W, r, pl)
        qc = _score(E, W_ref, c, pl)[0]; qr = _score(E, W_ref, r, pl)[0]
        hs.append((sc - qc) - (sr - qr)); rc.append(beta * (sc - qc)); rr.append(beta * (sr - qr))
        grads.append(gc - gr)
    h = np.array(hs)
    loss = float(np.logaddexp(0.0, -beta * h).mean())
    return h, loss, rc, rr, grads

def dpo_step(pairs, E, W, W_ref, beta, lr):
    E, W, W_ref = (np.asarray(a, float) for a in (E, W, W_ref))
    h, loss0, _, _, grads = _stats(pairs, E, W, W_ref, beta)
    n = len(pairs)
    coef = -beta / (1 + np.exp(beta * h)) / n
    dW = sum(c * g for c, g in zip(coef, grads))
    W2 = W - lr * dW
    h2, loss1, rc, rr, _ = _stats(pairs, E, W2, W_ref, beta)
    return {"loss_before": loss0, "loss_after": loss1, "margin_before": float((beta * h).mean()),
            "margin_after": float((beta * h2).mean()), "accuracy_after": float((h2 > 0).mean()),
            "chosen_reward_after": float(np.mean(rc)), "rejected_reward_after": float(np.mean(rr))}
'''

HINTS = [
    'Score a reply by summing log-probabilities of its own tokens only; the shared prompt cancels in h anyway.',
    'sigma(-beta h) is the weight DPO gives a pair: pairs the model already ranks correctly by a wide margin barely contribute.',
    'Check the gradient against finite differences on one pair before running the batch.',
]

def _case(seed, V, d, n, plen, rlen, start_at_ref, bias):
    g = np.random.default_rng(seed)
    E = g.normal(size=(V, d))
    W_ref = 0.5 * g.normal(size=(d, V))
    W = W_ref.copy() if start_at_ref else W_ref + 0.3 * g.normal(size=(d, V))
    pairs = []
    for _ in range(n):
        p = g.integers(0, V, plen).tolist()
        good = (p[-1] + 1 + np.arange(rlen)) % V if bias else g.integers(0, V, rlen)
        pairs.append((p + list(map(int, good)), p + g.integers(0, V, rlen).tolist(), plen))
    return (pairs, E, W, W_ref)

TESTS = [
    _case(1, 6, 4, 4, 3, 3, True, True) + (0.1, 5.0),
    _case(2, 8, 4, 8, 2, 4, True, True) + (0.5, 1.0),
    _case(3, 8, 6, 8, 2, 4, False, False) + (0.1, 2.0),
    _case(4, 10, 8, 16, 4, 5, True, True) + (1.0, 0.2),
    ([([0, 1, 2], [0, 1, 0], 2)], np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), 0.5, 1.0),
]

EXPLANATION = r"""
## The pieces in one step
Scoring whole replies with the model, the DPO loss and its implicit rewards, and the metrics to watch all come together here. The only new part is the gradient, and it has a clean form: each pair pushes $W$ along the difference between the chosen reply's log-likelihood gradient and the rejected one's, weighted by $\beta\,\sigma(-\beta h)$.

## What the tests show
- Starting from the reference model (every test except the third), every margin is 0 and the loss is exactly $\log 2$. One step makes the margin positive, mostly by raising the chosen reward.
- The third test starts away from the reference, with random "chosen" replies. Both rewards end up negative and the margin grows only because the rejected reward is lower: the pattern from the metrics step. The loss still drops, but with random preferences that is memorisation, not alignment.

## Why no reward model
The DPO loss is the reward-model loss with the reward written as $\beta\log(\pi/\pi_{ref})$. Optimising it fits a reward model and the policy that is optimal for it, in one supervised step, without sampling.

## Interview follow-ups
- DPO is trained on fixed pairs collected from another model. What goes wrong as the policy moves away from that data, and what do online or iterative variants change?
"""
