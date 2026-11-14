---
id: "01.03"
course: "deep-learning"
module: "01"
slug: "loss-functions-for-deep-learning"
title: "Loss functions for deep learning"
keyIdeas: ["Most losses are negative log-likelihoods: cross-entropy assumes a categorical output, MSE a Gaussian with fixed variance, so the loss encodes what noise you believe in.", "Softmax plus cross-entropy has the gradient $p - y$ with respect to the logits; compute it from logits with log-sum-exp, never by taking the log of probabilities.", "MSE's gradient grows linearly with the error, so outliers dominate; Huber loss switches to a constant-magnitude gradient beyond a threshold $\\delta$.", "Focal loss multiplies cross-entropy by $(1-p_t)^\\gamma$ to down-weight easy examples; margin losses (hinge, triplet) only push until a gap is reached and then stop."]
question: "You train a classifier on 1% positives and 99% negatives with plain binary cross-entropy. It reaches 99% accuracy but almost never predicts the positive class. Explain, using gradients, why plain BCE behaves this way and how focal loss and class weighting each change the gradient contributions."
answer: "Each example contributes a gradient of size $|p - y|$ on its logit. Early on the network learns to output a low probability for everything; then 99% of examples are easy negatives with small but non-zero gradients, and their sum still outweighs the few positives, so the optimiser keeps refining the negatives and accepts misclassifying positives. Accuracy looks great because predicting all-negative is 99% correct. Class weighting multiplies the positive examples' loss (and gradient) by a factor like 99, rebalancing the total gradient mass between classes regardless of difficulty. Focal loss instead scales each example by $(1-p_t)^\\gamma$: well-classified negatives with $p_t \\approx 0.95$ are scaled by about $0.0025$ at $\\gamma=2$, while misclassified positives keep most of their gradient. It reweights by difficulty, which handles imbalance indirectly because easy examples are mostly the majority class. In practice both are combined (the $\\alpha_t$ term) and the threshold is tuned on a validation set."
sources: [{"title": "Lin et al.: Focal Loss for Dense Object Detection", "url": "https://arxiv.org/abs/1708.02002", "note": "Sections 3 and 5: the focal term, why easy negatives dominate, and the $\\gamma$ ablation."}, {"title": "PyTorch docs: CrossEntropyLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html", "note": "Takes logits, supports class weights, ignore_index and label smoothing."}, {"title": "PyTorch docs: HuberLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html", "note": "Exact definition and its relation to SmoothL1Loss."}, {"title": "Deep Learning (Goodfellow et al.), Chapter 6.2", "url": "https://www.deeplearningbook.org/contents/mlp.html", "note": "Cost functions as maximum likelihood and why saturating outputs need log-likelihood losses."}]
practice: ["cross-entropy-logits", "softmax-cross-entropy-gradient", "log-loss-binary-cross-entropy", "loss-functions", "sigmoid-focal-loss"]
tags: ["cross-entropy", "Huber", "focal loss"]
tools: ["NumPy", "PyTorch"]
---

## A loss is a modelling choice

The loss is the only signal the network gets about what you want. Most standard losses are the **negative log-likelihood** of the data under an assumed output distribution, which makes the choice less arbitrary than it looks:

- Categorical output, $p = \mathrm{softmax}(z)$: NLL is **cross-entropy**.
- Bernoulli output, $p = \sigma(z)$: NLL is **binary cross-entropy**.
- Gaussian output with fixed variance: NLL is **mean squared error** up to constants.
- Laplace output: NLL is **mean absolute error**.

So picking MSE means "I believe errors are Gaussian"; if they are heavy-tailed, the loss will over-react to outliers. The ML course covers logistic regression and the classical metrics; AI Research covers entropy and KL divergence. This lesson focuses on how the losses behave inside a deep network's training loop.

## Cross-entropy from logits

For $C$ classes, logits $z \in \mathbb{R}^C$ and true class $y$:

$$\mathcal{L} = -\log \frac{e^{z_y}}{\sum_j e^{z_j}} = -z_y + \log\sum_j e^{z_j}$$

$$\frac{\partial \mathcal{L}}{\partial z_j} = p_j - \mathbb{1}[j = y]$$

The gradient is "predicted probability minus target", bounded in $[-1, 1]$ per logit. This clean form is why softmax and cross-entropy are always fused. The stable computation uses log-sum-exp with the max subtracted:

```python
import numpy as np

def cross_entropy(logits, y):          # logits (B, C), y (B,) int labels
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    loss = lse - logits[np.arange(len(y)), y]
    p = np.exp(logits - lse[:, None])
    grad = p.copy(); grad[np.arange(len(y)), y] -= 1
    return loss.mean(), grad / len(y)
```

Binary cross-entropy has the same trick: $\mathcal{L} = \max(z,0) - zy + \log(1 + e^{-|z|})$. This is why PyTorch's `BCEWithLogitsLoss` and `CrossEntropyLoss` take raw logits. Feeding them probabilities (or applying softmax twice) is a common silent bug: training still runs but gradients are badly scaled.

**Label smoothing** replaces the one-hot target with $1-\varepsilon$ on the true class and $\varepsilon/(C-1)$ elsewhere (one common convention). It stops the network driving logits to infinity to reach probability 1, which improves calibration.

<div class="ex">
<div class="lab">Worked example</div>

**one cross-entropy step by hand**

Logits $z = [2.0, 1.0, 0.1]$, true class 0. Exponentials: 7.389, 2.718, 1.105; sum 11.212.

Probabilities: $p = [0.659, 0.242, 0.099]$. Loss $= -\ln 0.659 = 0.417$.

Gradient on logits: $[0.659 - 1, 0.242, 0.099] = [-0.341, 0.242, 0.099]$. The gradient sums to zero: raising the true logit and lowering the others are the same move.

</div>

## Regression: MSE, MAE, Huber

MSE $= \frac{1}{2}(\hat y - y)^2$ has gradient $\hat y - y$, linear in the error. One target that is wrong by 100 contributes a gradient 100× larger than a typical example, so a few outliers steer the whole batch. MAE has gradient $\pm 1$, robust but non-smooth at zero and slow to fine-tune. **Huber** blends them:

$$\mathcal{L}_\delta(r) = \begin{cases} \tfrac{1}{2} r^2 & |r| \le \delta \\ \delta(|r| - \tfrac{1}{2}\delta) & |r| > \delta \end{cases}$$

Quadratic near zero, linear beyond $\delta$, so the gradient magnitude is capped at $\delta$. DQN uses it on TD errors for exactly this reason (Module 08); detection uses smooth-L1 (Huber with $\delta = 1$) for box regression.

## Focal and margin losses

**Focal loss** targets class imbalance in dense detection, where a single image has around 100k candidate boxes and almost all are easy background:

$$\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log p_t$$

with $p_t$ the probability of the true class and typically $\gamma = 2$. An easy example at $p_t = 0.9$ has its loss scaled by $0.01$; a hard one at $p_t = 0.1$ by $0.81$. The thousands of easy negatives stop drowning out the rare positives.

**Margin losses** don't model probabilities at all. They push until a required gap is reached and then contribute zero:

- **Hinge** (SVM): $\max(0, 1 - y \cdot s)$ for labels $\pm 1$.
- **Triplet**: $\max(0, d(a, p) - d(a, n) + m)$, pulling an anchor closer to a positive than to a negative by margin $m$; used for embeddings and face recognition.
- **Contrastive / InfoNCE**: cross-entropy over similarities, one positive among many negatives (CLIP); really a softmax loss in disguise.

| Loss | Gradient size for a bad example | Typical use |
| --- | --- | --- |
| Cross-entropy | bounded, about 1 per logit | classification, language models |
| MSE | grows with error | clean regression |
| Huber | capped at $\delta$ | noisy regression, box offsets, TD errors |
| Focal | CE scaled by $(1-p_t)^\gamma$ | dense detection, extreme imbalance |
| Hinge / triplet | constant until margin met, then 0 | SVMs, metric learning |

~~~dg-flow Where the loss sits: it is the root of the computational graph, and its gradient is the first signal backprop sends.
Logits -(softmax)-> Probabilities -(−log p_y)-> *Scalar loss -(∂L/∂z = p − y)-> Backward pass
~~~

<div class="take">
<div class="lab">Key takeaway</div>

choose the loss as a statement about your output distribution and your data's noise, compute it from logits in its stable form, and reach for Huber, focal or margin losses when outliers, imbalance or ranking make plain NLL misbehave.

</div>
