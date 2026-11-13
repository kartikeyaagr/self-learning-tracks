import numpy as np

META = dict(title='Sigmoid focal loss', tags=['Loss Functions', 'Object Detection'], diff='Medium', runtime='numpy',
            lessons=[], related=['log-loss-binary-cross-entropy', 'cross-entropy-logits', 'dice-loss'])

STATEMENT = r"""
A dense detector scores tens of thousands of anchors per image and almost all of them are easy background. With plain binary cross-entropy those easy negatives dominate the loss. **Focal loss** scales each term by $(1 - p_t)^\gamma$, which is near 0 for confident, correct predictions and near 1 for mistakes.

With logit $z$, target $t \in [0, 1]$ and $p = \sigma(z)$:

$$p_t = p\,t + (1 - p)(1 - t), \qquad \mathrm{CE} = -\big[t \log p + (1 - t)\log(1 - p)\big]$$
$$\mathrm{FL} = \alpha_t\,(1 - p_t)^{\gamma}\,\mathrm{CE}, \qquad \alpha_t = \alpha\, t + (1 - \alpha)(1 - t)$$

Write `sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean")`.

## Input
- `logits`: float array of any shape; `targets`: float array of the same shape with values in `[0, 1]` (usually 0/1).
- `alpha`: a float in `[0, 1]`, or `None` to disable the weighting (use $\alpha_t = 1$).
- `gamma`: float `>= 0`. With `gamma = 0` and `alpha = None` this is plain BCE.
- `reduction`: `"none"` (return the elementwise loss array, same shape), `"mean"` or `"sum"` (return a Python float).

## Numerical stability
Logits can be large, e.g. $\pm 100$. Compute CE directly from the logit as $\max(z, 0) - z t + \log(1 + e^{-|z|})$, and compute $\sigma(z)$ without overflow. The result must be finite.
"""

STARTER = '''import numpy as np

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """Elementwise focal loss on logits; reduction in {"none", "mean", "sum"}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    e = np.exp(-np.abs(z))                              # in (0, 1], never overflows
    p = np.where(z >= 0, 1 / (1 + e), e / (1 + e))      # stable sigmoid
    ce = np.maximum(z, 0) - z * t + np.log1p(e)         # BCE with logits
    p_t = p * t + (1 - p) * (1 - t)
    loss = ce * (1 - p_t) ** gamma
    if alpha is not None:
        loss = loss * (alpha * t + (1 - alpha) * (1 - t))
    if reduction == "none":
        return loss
    return float(loss.sum() if reduction == "sum" else loss.mean())
'''

HINTS = [
    'Start from a numerically stable binary cross-entropy on logits, then multiply by the modulating factor and the alpha weight.',
    'Stable pieces: `e = exp(-|z|)`; `sigmoid(z) = 1/(1+e)` for `z >= 0` and `e/(1+e)` otherwise; `CE = max(z, 0) - z*t + log1p(e)`.',
    '`p_t = p*t + (1-p)*(1-t)`, `loss = CE * (1 - p_t)**gamma`, times `alpha*t + (1-alpha)*(1-t)` unless `alpha is None`; then reduce.',
]

_r = np.random.default_rng(28)
TESTS = [
    (np.array([0.0, 2.0, -2.0]), np.array([1.0, 1.0, 0.0]), 0.25, 2.0, "none"),
    (np.array([0.0, 0.0]), np.array([1.0, 0.0]), None, 0.0, "mean"),
    (np.array([100.0, -100.0, 100.0, -100.0]), np.array([1.0, 0.0, 0.0, 1.0]), 0.25, 2.0, "none"),
    (_r.normal(size=(4, 5)) * 3, (_r.uniform(size=(4, 5)) < 0.2).astype(float)),
    (_r.normal(size=(4, 5)) * 3, (_r.uniform(size=(4, 5)) < 0.2).astype(float), 0.5, 0.0, "sum"),
    (_r.normal(size=10), _r.uniform(size=10), None, 1.5, "mean"),
    (np.array([3.0]), np.array([1.0]), 0.75, 5.0, "sum"),
]

EXPLANATION = r"""
## What the two knobs do
- $(1 - p_t)^\gamma$ down-weights easy examples. With $\gamma = 2$, an example classified with $p_t = 0.9$ contributes 100 times less than under BCE, while a badly misclassified one ($p_t = 0.1$) keeps 81% of its weight.
- $\alpha$ balances positives against negatives. In RetinaNet $\alpha = 0.25$ for positives is counter-intuitive, but the $\gamma$ term already suppresses the many easy negatives so much that positives need less extra weight.

## Stability
$\log \sigma(z)$ for $z = -100$ underflows to $\log 0$ if computed naively. The rewritten CE never takes the log of something small: $\log(1 + e^{-|z|})$ stays in $[0, \log 2]$. Test 3 has confident wrong predictions whose loss is about 100 times $\alpha_t$.

## Complexity
$O(n)$ elementwise. In detection the sum is usually divided by the number of **positive anchors**, not the number of anchors, so the loss scale doesn't depend on how much background an image has.

## Common bugs
- Computing `np.log(sigmoid(z))`: `-inf` or `nan` for large logits.
- Applying `alpha` to both classes instead of `alpha` for positives and `1 - alpha` for negatives.
- Using `mean` over all anchors in a detector, which makes the loss tiny and the gradients vanish.

## Interview follow-ups
- Derive the gradient with respect to $z$ for $t = 1$. How does it compare to BCE's $p - 1$?
- How would you initialise the classifier's bias so training starts stable? (Set it to $-\log((1 - \pi)/\pi)$ with prior $\pi \approx 0.01$, so every anchor starts as confident background.)
"""
