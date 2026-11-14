import numpy as np

META = dict(title='The IPO loss', tags=['Language Models', 'Loss Functions'], diff='Easy', runtime='numpy',
            related=['dpo-closed-form-loss', 'dpo-reward-accuracy-margin'])

STATEMENT = r"""
DPO's logistic loss keeps rewarding a larger margin forever, so on separable data it pushes the log-ratio towards infinity. IPO (Azar et al., 2023) replaces it with a squared loss towards a fixed target.

Write `ipo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, tau)`:

$$h = (\log\pi_\theta(y_c) - \log\pi_{ref}(y_c)) - (\log\pi_\theta(y_r) - \log\pi_{ref}(y_r)), \qquad L = \text{mean}\Big(\big(h - \tfrac{1}{2\tau}\big)^2\Big).$$

Return a dict with `loss` and `h` (the list of per-pair values).
"""

STARTER = '''import numpy as np

def ipo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, tau):
    """IPO: regress the log-ratio margin onto 1 / (2 tau)."""
    h = (np.asarray(pol_chosen) - np.asarray(ref_chosen)) - (np.asarray(pol_rejected) - np.asarray(ref_rejected))
    return {"loss": float(np.mean((h - tau) ** 2)), "h": h.tolist()}
'''

SOLUTION = '''import numpy as np

def ipo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, tau):
    h = (np.asarray(pol_chosen, float) - np.asarray(ref_chosen, float)) - (np.asarray(pol_rejected, float) - np.asarray(ref_rejected, float))
    return {"loss": float(np.mean((h - 1.0 / (2.0 * tau)) ** 2)), "h": h.tolist()}
'''

HINTS = [
    'The target margin is 1 / (2 tau), so a smaller tau asks for a larger margin.',
    'Unlike DPO, overshooting the target is penalised too: the loss is symmetric around it.',
]

TESTS = [
    ([0.0], [0.0], [0.0], [0.0], 0.5),
    ([-9.0], [-14.0], [-10.0], [-12.0], 0.5),
    ([-9.0], [-14.0], [-10.0], [-12.0], 0.2),
    ([-5.0, -8.0, -20.0], [-30.0, -9.0, -21.0], [-10.0, -8.0, -20.0], [-12.0, -9.0, -20.0], 0.25),
    (list(-np.linspace(10, 40, 6)), list(-np.linspace(15, 50, 6)), list(-np.linspace(11, 41, 6)), list(-np.linspace(14, 47, 6)), 1.0),
]

EXPLANATION = r"""
## Why DPO can run away
The DPO loss $-\log\sigma(\beta h)$ is never zero, so the gradient always says "more margin". If each preferred reply always beats its rejected one, the cheapest way to increase the margin is often to drive the rejected reply's probability towards zero, and the policy drifts far from the reference even with a KL coefficient. The $\beta$ that was meant to keep it close loses its grip.

## What IPO changes
Squared error to a target $1/(2\tau)$ has a minimum. Pairs whose margin already exceeds the target get pushed back (the first pair in the fourth test, with a margin of 23, is far past the target of 2). This keeps the regularisation that the KL term was supposed to provide.

## Interview follow-ups
- IPO treats every pair as equally strong evidence. When would you want a loss that scales the target by how confident the annotators were?
"""
