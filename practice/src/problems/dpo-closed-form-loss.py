import numpy as np

META = dict(title='DPO loss from token log-probabilities', tags=['Reinforcement Learning', 'Language Models', 'Loss Functions'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.04', 'air-fanout-v1:01.12'],
            related=['bradley-terry-reward-model-loss', 'ppo-with-reference-policy-kl', 'kl-divergence-discrete-and-continuous'])

STATEMENT = r"""
Direct Preference Optimization skips the reward model. The KL-regularised RLHF objective has a closed-form optimum in which the reward is $\beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ up to a constant, so the Bradley-Terry loss can be written directly in terms of the policy:

$$z_i = \beta\Big[\big(\log\pi(y_c) - \log\pi_{\text{ref}}(y_c)\big) - \big(\log\pi(y_r) - \log\pi_{\text{ref}}(y_r)\big)\Big]$$
$$L = \frac1B \sum_i \Big[-(1 - \varepsilon) \log\sigma(z_i) - \varepsilon \log\sigma(-z_i)\Big]$$

where $\varepsilon$ is optional label smoothing (the probability a label is flipped). Sequence log-probabilities are sums of per-token log-probabilities over **response** tokens only.

Write `dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, mask_chosen, mask_rejected, beta=0.1, label_smoothing=0.0)` and return a dict with:

| key | value |
| --- | --- |
| `"loss"` | float, $L$ |
| `"chosen_rewards"` | float `(B,)`, $\beta(\log\pi(y_c) - \log\pi_{\text{ref}}(y_c))$ |
| `"rejected_rewards"` | float `(B,)`, $\beta(\log\pi(y_r) - \log\pi_{\text{ref}}(y_r))$ |
| `"accuracy"` | float, fraction of pairs with chosen reward strictly greater than rejected |
| `"grad_chosen"` | float `(B,)`, $\partial L / \partial \log\pi(y_c)$ (the policy's **sequence** log-prob) |
| `"grad_rejected"` | float `(B,)`, $\partial L / \partial \log\pi(y_r)$ |

## Inputs
- `pol_chosen`, `ref_chosen`, `mask_chosen`: `(B, Tc)`; `pol_rejected`, `ref_rejected`, `mask_rejected`: `(B, Tr)`. The first four hold per-token log-probabilities; masks are 0/1 with 1 on response tokens. Prompt and padding positions may hold any value and must be ignored.

Compute $\log\sigma$ stably.
"""

STARTER = '''import numpy as np

def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, mask_chosen, mask_rejected,
             beta=0.1, label_smoothing=0.0):
    """Token log-probs (B, T) plus response masks. Return dict (see statement)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, mask_chosen, mask_rejected,
             beta=0.1, label_smoothing=0.0):
    seq = lambda lp, m: (np.asarray(lp, dtype=float) * m).sum(axis=1)   # sum over response tokens
    mc, mr = np.asarray(mask_chosen, dtype=float), np.asarray(mask_rejected, dtype=float)
    rew_c = beta * (seq(pol_chosen, mc) - seq(ref_chosen, mc))
    rew_r = beta * (seq(pol_rejected, mr) - seq(ref_rejected, mr))
    z = rew_c - rew_r
    eps, B = label_smoothing, len(z)
    # -log sigmoid(z) = logaddexp(0, -z)
    loss = ((1 - eps) * np.logaddexp(0.0, -z) + eps * np.logaddexp(0.0, z)).mean()
    sig = lambda x: np.exp(-np.logaddexp(0.0, -x))                      # stable sigmoid
    dL_dz = (-(1 - eps) * sig(-z) + eps * sig(z)) / B
    return {"loss": float(loss), "chosen_rewards": rew_c, "rejected_rewards": rew_r,
            "accuracy": float((rew_c > rew_r).mean()),
            "grad_chosen": beta * dL_dz, "grad_rejected": -beta * dL_dz}
'''

HINTS = [
    'Reduce to sequences first: multiply each token log-prob array by its mask and sum over the time axis. Masked-out entries may be garbage, so multiply (or use `where`), never just slice.',
    'The implicit rewards are $\\beta$ times policy minus reference sequence log-probs; $z$ is their difference. Use `np.logaddexp(0, -z)` for $-\\log\\sigma(z)$.',
    '$dL/dz = [-(1-\\varepsilon)\\sigma(-z) + \\varepsilon\\,\\sigma(z)]/B$. $z$ depends on the chosen log-prob with coefficient $+\\beta$ and on the rejected one with $-\\beta$.',
]

_r = np.random.default_rng(251)
_B, _Tc, _Tr = 4, 6, 5
def _lp(shape):
    return -np.abs(_r.normal(size=shape)) - 0.1
PC, RC = _lp((_B, _Tc)), _lp((_B, _Tc))
PR, RRf = _lp((_B, _Tr)), _lp((_B, _Tr))
MC = (np.arange(_Tc) >= np.array([1, 2, 0, 3])[:, None]).astype(int)          # prompt tokens masked at the front
MR = ((np.arange(_Tr) >= 1) & (np.arange(_Tr) < np.array([5, 3, 4, 2])[:, None])).astype(int)
TESTS = [
    (np.array([[-1.0, -2.0]]), np.array([[-3.0, -1.0]]), np.array([[-1.5, -2.0]]), np.array([[-2.0, -1.0]]),
     np.array([[1, 1]]), np.array([[1, 1]]), 1.0),
    (np.array([[-1.0, -2.0]]), np.array([[-3.0, -1.0]]), np.array([[-1.0, -2.0]]), np.array([[-3.0, -1.0]]),
     np.array([[1, 1]]), np.array([[1, 1]]), 0.1),
    (np.array([[-50.0, -1.0, -1.0]]), np.array([[-50.0, -2.0]]), np.array([[0.0, -2.0, -1.0]]), np.array([[0.0, -1.0]]),
     np.array([[0, 1, 1]]), np.array([[0, 1]]), 0.5),
    (np.array([[-1.0]]), np.array([[-1.0]]), np.array([[-1.0]]), np.array([[-1.0]]), np.array([[1]]), np.array([[1]]), 0.1, 0.1),
    (np.array([[0.0]]), np.array([[-2000.0]]), np.array([[-2000.0]]), np.array([[0.0]]), np.array([[1]]), np.array([[1]]), 1.0),
    (PC, PR, RC, RRf, MC, MR, 0.1),
    (PC, PR, RC, RRf, MC, MR, 2.0, 0.2),
]

EXPLANATION = r"""
## Where the closed form comes from
The RLHF objective $\max_\pi \mathbb{E}[r(x,y)] - \beta\,\mathrm{KL}(\pi \Vert \pi_{\text{ref}})$ is maximised by $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\, e^{r(x,y)/\beta}$. Solving for $r$ gives $r = \beta \log\frac{\pi^*}{\pi_{\text{ref}}} + \beta\log Z(x)$. The partition function $Z(x)$ depends only on the prompt, so it cancels in the Bradley-Terry difference between two responses to the same prompt. What remains is a logistic loss on log-ratio differences: preference learning with no reward model and no sampling.

## Reading the gradient
$\partial L/\partial\log\pi(y_c) = -\beta\,\sigma(-z)/B$ (with $\varepsilon = 0$): raise the chosen response's log-prob, lower the rejected one's, weighted by how *wrong* the implicit reward currently is. Pairs the model already ranks confidently ($z \gg 0$) contribute almost nothing. Test 2 starts at the reference policy: $z = 0$, loss $\log 2$, and the weight is exactly $1/2$.

## Masks and stability
Test 3 puts $-50$ in a masked prompt position that must not count. Test 5 has $z = 4000$: the loss is $\approx 0$, finite, and the gradient is $\approx 0$.

## Label smoothing
With $\varepsilon > 0$ the loss is minimised at a finite $z$ ($\sigma(z) = 1 - \varepsilon$) instead of $z \to \infty$, which limits how far the policy is pushed on possibly mislabelled pairs (test 4: at $z = 0$ the two terms partly cancel in the gradient).

## Complexity
$O(B(T_c + T_r))$.

## Common bugs
- Averaging token log-probs instead of summing (length-normalised variants exist but are a different loss).
- Counting prompt tokens in the sums.
- Flipping the sign of the reference terms.

## Interview follow-ups
- DPO tends to lower the log-probability of *both* responses during training. Why can that happen while the loss still improves?
- How do IPO or KTO change the objective, and what problem does each address?
"""
