import numpy as np

META = dict(title='GRPO group advantages: three scalings and degenerate groups', tags=['Reinforcement Learning', 'Language Models'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.05'],
            related=['grpo-group-relative-advantage', 'rloo-reinforce-leave-one-out', 'compare-sequence-and-token-length-normalization'])

STATEMENT = r"""
GRPO replaces a learned critic with a statistic of the **group**: several responses are sampled per prompt, scored by a verifier, and each response's advantage is its reward minus the group mean. What to divide by is a real design choice, and groups in which every response got the same reward carry no learning signal at all.

Write `group_advantages(rewards, group_size, scale, eps=1e-6)` and return a dict.

## Inputs
- `rewards`: float64 `(N,)`, laid out group by group: responses `0 .. G-1` belong to prompt 0, the next `G` to prompt 1, and so on. $N$ is a multiple of $G$ = `group_size` $\ge 2$.
- `scale`: one of
  - `"group"`: divide by the group's **population** standard deviation (`ddof=0`) plus `eps` (original GRPO);
  - `"batch"`: divide by the population standard deviation of **all** $N$ rewards plus `eps`;
  - `"none"`: no division (mean-centring only, as in Dr. GRPO).

## Rules
1. $A_i = r_i - \bar r_{g(i)}$, then apply the scaling.
2. A group is **degenerate** if its maximum reward equals its minimum (compare exactly; don't test the standard deviation against 0). Its advantages are exactly `0.0`.

## Output
| key | value |
|---|---|
| `"advantages"` | float64 `(N,)` |
| `"degenerate"` | bool `(N / G,)`, one flag per group |
| `"keep"` | bool `(N,)`, `False` for responses in degenerate groups (dynamic sampling filters them out) |
| `"frac_degenerate"` | float |

Use reshapes, not Python loops over groups.
"""

STARTER = '''import numpy as np

def group_advantages(rewards, group_size, scale, eps=1e-6):
    """Return a dict: advantages (N,), degenerate (P,), keep (N,), frac_degenerate."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def group_advantages(rewards, group_size, scale, eps=1e-6):
    r = np.asarray(rewards, dtype=float)
    g = r.reshape(-1, group_size)                         # (P, G): one row per prompt
    adv = g - g.mean(axis=1, keepdims=True)
    if scale == "group":
        adv = adv / (g.std(axis=1, keepdims=True) + eps)
    elif scale == "batch":
        adv = adv / (r.std() + eps)
    degenerate = g.max(axis=1) == g.min(axis=1)           # exact test: a float std can be 1e-17, not 0
    adv[degenerate] = 0.0
    return {"advantages": adv.reshape(-1), "degenerate": degenerate,
            "keep": np.repeat(~degenerate, group_size), "frac_degenerate": float(degenerate.mean())}
'''

HINTS = [
    'Reshape the rewards to `(N // G, G)`; then group means and standard deviations are reductions along `axis=1` with `keepdims=True`.',
    'For `"batch"` scaling, the denominator is one scalar computed over all rewards, but the centring still uses the **group** mean.',
    'Detect degenerate groups with `max == min` and overwrite their rows with 0: the mean of three copies of 0.1 is not exactly 0.1 in floating point, so centring alone can leave residues of $10^{-17}$, and dividing those by `eps` blows them up.',
]

_r = np.random.default_rng(31)
TESTS = [
    (np.array([1.0, 0.0, 0.0, 1.0]), 2, 'group'),
    (np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0]), 3, 'none'),
    (np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0]), 3, 'batch'),
    (np.array([0.1, 0.1, 0.1, 0.7, 0.2, 0.4]), 3, 'group'),
    ((_r.random(32) < 0.4).astype(float), 8, 'group'),
    (_r.normal(size=24), 4, 'batch'),
    (np.array([5.0, 5.0, 5.0, 5.0]), 2, 'batch'),
    (np.round(_r.random(40), 1), 5, 'group', 1e-4),
]

EXPLANATION = r"""
## What each scaling does
- **group**: every prompt contributes advantages of unit scale, so a prompt where the model is 1-for-16 gets the same push as one where it is 8-for-16. That up-weights near-solved and near-impossible prompts, where the standard deviation is small: a difficulty bias.
- **batch**: one scale for the whole batch keeps the relative size of signals across prompts and still fixes the overall magnitude for the learning rate.
- **none**: plain mean-centring, an unbiased policy-gradient baseline. Dr. GRPO argues the division (along with per-sequence length averaging) is what biases GRPO.

Tests 2 and 3 use the same rewards with different scalings.

## Degenerate groups
If all $G$ responses got the same reward (all correct or all wrong), the centred advantages are zero, so those samples cost a full forward and backward pass for no gradient. As training progresses, more prompts become all-correct; DAPO's *dynamic sampling* filters them out and samples more prompts to keep the effective batch size. The exact `max == min` test matters: in test 4, `[0.1, 0.1, 0.1]` has a floating-point mean slightly off 0.1, and with `"group"` scaling the tiny residues divided by `eps` would become advantages of order $10^{-11}$ or larger instead of exactly 0.

## Complexity
$O(N)$.

## Common bugs
- Sample standard deviation (`ddof=1`) where population is specified, or vice versa: both appear in published code, so read the spec.
- With `"batch"`, centring by the batch mean instead of the group mean, which turns easy prompts into large positive advantages.
- Forgetting `eps`, so a degenerate group divides 0 by 0.

## Interview follow-ups
- With binary rewards and group size $G$, what is the largest possible advantage under `"group"` scaling, and for which success count does it occur?
- How does the leave-one-out baseline (RLOO) differ from subtracting the group mean, and why is it unbiased?
"""
