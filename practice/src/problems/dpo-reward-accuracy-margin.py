import numpy as np

META = dict(title='DPO training metrics: reward accuracy and margin', tags=['Language Models', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['dpo-closed-form-loss'])

STATEMENT = r"""
DPO defines an implicit reward for a reply $y$: $\hat r(y) = \beta\,(\log\pi_\theta(y) - \log\pi_{ref}(y))$, with sequence log-probabilities.

Write `dpo_metrics(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta)` for a batch of preference pairs. Return a dict:

- `chosen_reward`, `rejected_reward`: batch means of $\hat r$ for the chosen and rejected replies;
- `margin`: mean of $\hat r(y_c) - \hat r(y_r)$;
- `accuracy`: fraction of pairs with $\hat r(y_c) > \hat r(y_r)$ (strictly);
- `loss`: mean of $-\log\sigma(\hat r(y_c) - \hat r(y_r))$, computed stably.
"""

STARTER = '''import numpy as np

def dpo_metrics(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta):
    """Implicit rewards, margin, accuracy and loss for a batch of DPO pairs."""
    c = np.asarray(pol_chosen) - np.asarray(pol_rejected)
    return {"chosen_reward": 0.0, "rejected_reward": 0.0, "margin": float(c.mean()),
            "accuracy": float((c > 0).mean()), "loss": float(-np.log(1 / (1 + np.exp(-c))).mean())}
'''

SOLUTION = '''import numpy as np

def dpo_metrics(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta):
    rc = beta * (np.asarray(pol_chosen, float) - np.asarray(ref_chosen, float))
    rr = beta * (np.asarray(pol_rejected, float) - np.asarray(ref_rejected, float))
    m = rc - rr
    loss = np.logaddexp(0.0, -m).mean()
    return {"chosen_reward": float(rc.mean()), "rejected_reward": float(rr.mean()), "margin": float(m.mean()),
            "accuracy": float((m > 0).mean()), "loss": float(loss)}
'''

HINTS = [
    'The rewards are relative to the reference model; a pair the reference already prefers is not a win for the policy.',
    '-log sigmoid(m) equals logaddexp(0, -m), which never overflows.',
    'At the start of training the policy equals the reference, so every reward is 0, accuracy is 0 and the loss is log 2.',
]

def _case(seed, n, drift_c, drift_r):
    g = np.random.default_rng(seed)
    rc, rr = -g.uniform(20, 60, n), -g.uniform(20, 60, n)
    return (rc + drift_c + g.normal(size=n), rr + drift_r + g.normal(size=n), rc, rr)

TESTS = [
    ([-10.0, -20.0], [-12.0, -18.0], [-10.0, -20.0], [-12.0, -18.0], 0.1),
    ([-9.0], [-14.0], [-10.0], [-12.0], 0.1),
    _case(1, 16, -2.0, -8.0) + (0.1,),
    _case(2, 16, 1.0, -1.0) + (0.5,),
    ([-500.0], [-10.0], [-10.0], [-10.0], 1.0),
]

EXPLANATION = r"""
## What to watch
These are the numbers DPO training libraries log every step.

- **Accuracy** should rise from 0 (at the start every margin is exactly 0) towards 1 on the training set.
- **Margin** should grow. It is the quantity the loss pushes on.
- **Chosen and rejected rewards** tell you how the margin grows. A common and worrying pattern, seen in the third test, is that both go down: the policy lowers the probability of the chosen reply too, just less than the rejected one. The policy's probability mass then goes to replies that are in neither set, which can make it more verbose or strange.

## The last test
A huge negative margin gives a loss of about 490. A naive `-log(sigmoid(m))` returns infinity there; the stable form does not.

## Interview follow-ups
- Accuracy is near 1 on training pairs but held-out accuracy stays near 0.6. What does that suggest, and what would you change?
"""
