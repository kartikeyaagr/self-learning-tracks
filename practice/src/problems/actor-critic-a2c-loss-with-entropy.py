import numpy as np

META = dict(title='A2C loss with an entropy bonus, and its gradients', tags=['Reinforcement Learning', 'Loss Functions'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.02', 'air-fanout-v1:07.04'],
            related=['reinforce-with-baseline', 'gae-advantages', 'ppo-clipped-surrogate-objective'])

STATEMENT = r"""
Advantage actor-critic trains a policy head (logits) and a value head together with one combined loss over a batch of $B$ steps:

$$A_i = R_i - V_i \quad (\text{treated as a constant in the policy term})$$
$$L_\pi = -\frac1B \sum_i A_i \log \pi(a_i \mid s_i), \qquad L_V = \frac1B \sum_i \tfrac12 (R_i - V_i)^2, \qquad H = \frac1B \sum_i \Big(-\sum_a \pi_{i,a} \log \pi_{i,a}\Big)$$
$$L = L_\pi + c_V\, L_V - c_H\, H$$

The entropy bonus keeps the policy from collapsing to a deterministic one too early.

Write `a2c_loss(logits, actions, returns, values, vf_coef=0.5, ent_coef=0.01)` and return a dict with keys `"loss"`, `"policy_loss"`, `"value_loss"`, `"entropy"` (floats: $L, L_\pi, L_V, H$), `"grad_logits"` (float `(B, A)`, $\partial L / \partial\,$`logits`) and `"grad_values"` (float `(B,)`, $\partial L / \partial\,$`values`).

## Inputs
- `logits`: float `(B, A)`; `actions`: int `(B,)`.
- `returns`: float `(B,)`, the bootstrapped return targets $R_i$; `values`: float `(B,)`, the critic's $V_i$.

## Notes
- Use a stable log-softmax. Logits can be large, and a probability that underflows to 0 contributes 0 to the entropy.
- Because $A_i$ is a constant, `values` gets gradient only from $L_V$.
"""

STARTER = '''import numpy as np

def a2c_loss(logits, actions, returns, values, vf_coef=0.5, ent_coef=0.01):
    """Return dict: loss, policy_loss, value_loss, entropy, grad_logits (B, A), grad_values (B,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def a2c_loss(logits, actions, returns, values, vf_coef=0.5, ent_coef=0.01):
    z = np.asarray(logits, dtype=float)
    R, V = np.asarray(returns, dtype=float), np.asarray(values, dtype=float)
    B = len(R)
    rows = np.arange(B)
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))     # stable log-softmax
    p = np.exp(logp)
    adv = R - V                                                  # constant ("detached") in the policy term

    policy_loss = -(adv * logp[rows, actions]).mean()
    value_loss = 0.5 * ((R - V) ** 2).mean()
    H_i = -(p * logp).sum(axis=1)                                # per-row entropy; p*logp -> 0 when p underflows
    entropy = H_i.mean()

    # d(-A log pi_a)/dz = -A (onehot - p);   dH/dz_j = -p_j (log p_j + H)
    g_pi = p * adv[:, None]
    g_pi[rows, actions] -= adv
    g_ent = -p * (logp + H_i[:, None])
    grad_logits = (g_pi - ent_coef * g_ent) / B
    grad_values = -vf_coef * (R - V) / B
    return {"loss": float(policy_loss + vf_coef * value_loss - ent_coef * entropy),
            "policy_loss": float(policy_loss), "value_loss": float(value_loss), "entropy": float(entropy),
            "grad_logits": grad_logits, "grad_values": grad_values}
'''

HINTS = [
    'Compute a stable log-softmax once and derive `p = exp(logp)` from it; every term needs one or the other.',
    'Policy term: $\\partial(-A \\log \\pi_a)/\\partial z = -A(e_a - \\pi)$. Value term: $\\partial \\tfrac12(R-V)^2/\\partial V = -(R - V)$. Don\'t forget the $1/B$ and the coefficients.',
    'Entropy: with $H = -\\sum_j \\pi_j \\log \\pi_j$, the softmax chain rule gives $\\partial H / \\partial z_j = -\\pi_j(\\log \\pi_j + H)$. The loss has $-c_H H$, so subtract `ent_coef` times that.',
]

_r = np.random.default_rng(221)
ZR = _r.normal(size=(6, 4)) * 2
TESTS = [
    (np.zeros((1, 2)), np.array([0]), np.array([1.0]), np.array([0.0])),
    (np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), np.array([2, 1]), np.array([1.0, -1.0]), np.array([0.5, 0.0]), 1.0, 0.1),
    (np.array([[0.0, 0.0, 0.0]]), np.array([1]), np.array([2.0]), np.array([2.0]), 0.5, 0.0),
    (np.array([[800.0, 0.0]]), np.array([1]), np.array([1.0]), np.array([0.0]), 0.5, 0.01),
    (ZR, _r.integers(0, 4, 6), _r.normal(size=6), _r.normal(size=6)),
    (ZR, _r.integers(0, 4, 6), _r.normal(size=6), _r.normal(size=6), 0.25, 0.5),
]

EXPLANATION = r"""
## How the pieces interact
- The **policy term** is REINFORCE with the critic as baseline. It has to treat $A$ as a constant: if gradient flowed from it into the critic, the policy loss would train $V$ to inflate or deflate advantages instead of predicting returns.
- The **value term** is plain regression of $V$ toward the return. $c_V$ balances it against the policy term when the two heads share a trunk.
- The **entropy term** rewards spread-out policies. Its gradient $-\pi_j(\log\pi_j + H)$ pushes logits of below-average-log-probability actions up and above-average ones down, i.e. toward uniform. At the uniform policy it is exactly zero (test 3 has $c_H = 0$ and $A = 0$, so every gradient is zero).

## Stability
Test 4 has logits 800 apart. Plain `log(softmax)` gives $\log 0 = -\infty$ and `0 * -inf = nan` in the entropy. Log-softmax keeps $\log\pi_1 = -800$ finite and $\pi_1 \log\pi_1$ underflows cleanly to 0.

## Complexity
$O(BA)$ time and memory.

## Common bugs
- Letting the advantage carry gradient into `values` (getting `grad_values` from both terms).
- A sign slip on the entropy: the loss *subtracts* $c_H H$ because we want to maximise entropy.
- Using `sum` in one term and `mean` in another.

## Interview follow-ups
- A2C vs A3C: what did asynchronous workers buy, and why did synchronous batched A2C mostly replace it on GPUs?
- How would you add GAE advantages and PPO clipping to this loss to get PPO?
"""
