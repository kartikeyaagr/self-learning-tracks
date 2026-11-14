---
id: "01.02"
course: "deep-learning"
module: "01"
slug: "activation-functions"
title: "Activation functions"
keyIdeas: ["Sigmoid and tanh saturate: for large $|x|$ their derivative goes to zero, and sigmoid's derivative never exceeds 0.25, so stacked sigmoids shrink gradients geometrically.", "ReLU has gradient exactly 1 for positive inputs, which keeps deep gradients alive, but a unit whose input stays negative gets zero gradient forever (a dead unit).", "GELU, $x\\Phi(x)$, and SiLU, $x\\sigma(x)$, are smooth ReLU-like gates that let small negative values through; they are the defaults in Transformers.", "Write activations in numerically safe forms: compute sigmoid and softplus with branches or $\\log(1+e^{-|x|})$ so `exp` never overflows."]
question: "A 20-layer tanh network trains well at learning rate 1e-3 but at 1e-1 the loss stops improving within a few hundred steps, and inspecting activations shows most values at ±1. A ReLU version at 1e-1 stalls too, but its activations are mostly exactly 0. Explain both failures in terms of the activation derivatives, and suggest one fix for each."
answer: "Tanh: the large learning rate pushes weights large, so pre-activations land deep in the flat tails where $\\tanh' = 1 - \\tanh^2 \\approx 0$. Every saturated layer multiplies the backward signal by nearly zero, so gradients to early layers vanish and learning stalls even though the units are not dead in principle. Fix: lower learning rate, Xavier initialisation, or normalisation layers that keep pre-activations near zero. ReLU: a big update drove many units' pre-activations negative for all inputs; their derivative is exactly 0, so they get no gradient and never come back (dead units). The signal is not shrinking smoothly, it is cut off. Fix: lower learning rate or warmup, He initialisation, or Leaky ReLU/GELU so negative inputs keep a small gradient. Both cases show that the activation's derivative profile determines which way a too-large step breaks training."
sources: [{"title": "Hendrycks & Gimpel: Gaussian Error Linear Units (GELUs)", "url": "https://arxiv.org/abs/1606.08415", "note": "Definition of GELU, the tanh approximation and comparisons with ReLU and ELU."}, {"title": "Ramachandran, Zoph, Le: Searching for Activation Functions", "url": "https://arxiv.org/abs/1710.05941", "note": "Where Swish ($x\\sigma(\\beta x)$) came from, found by automated search."}, {"title": "CS231n: Neural Networks Part 1 (activation functions)", "url": "https://cs231n.github.io/neural-networks-1/", "note": "Sigmoid saturation, non-zero-centred outputs and dying ReLUs, concisely."}, {"title": "PyTorch docs: torch.nn.GELU", "url": "https://pytorch.org/docs/stable/generated/torch.nn.GELU.html", "note": "The exact and tanh-approximate forms as implemented."}]
practice: ["activation-functions", "derivatives-of-common-activation-functions", "sigmoid-stable", "continuity-of-activation-functions", "sigmoid-boundedness-via-squeeze-theorem"]
tags: ["activations", "saturation", "numerical stability"]
tools: ["NumPy"]
---

## What an activation is for

The previous lesson showed the nonlinearity is what makes depth meaningful. The choice of nonlinearity then decides two practical things: how gradients flow backward through many layers, and whether the forward computation stays finite in floating point. This lesson goes through the common choices in historical order, because each one fixes a failure of the last.

## Sigmoid and tanh: smooth but saturating

$$\sigma(x) = \frac{1}{1+e^{-x}}, \qquad \sigma'(x) = \sigma(x)(1-\sigma(x)) \le 0.25$$
$$\tanh(x) = 2\sigma(2x) - 1, \qquad \tanh'(x) = 1 - \tanh^2(x) \le 1$$

Both squash to a bounded range. Their problem is **saturation**: when $|x|$ is large the curve is flat, the derivative is near zero, and backprop multiplies the upstream gradient by that near-zero number. With sigmoid the best case is a factor of 0.25 per layer; ten layers gives at most $0.25^{10} \approx 10^{-6}$. Sigmoid is also not zero-centred: its outputs are all positive, so the gradients on a downstream neuron's weights all share one sign, which makes updates zig-zag. Tanh fixes the centring but still saturates.

Sigmoid remains the right tool where you need a probability or a gate: the output of a binary classifier, LSTM gates, attention-free gating like squeeze-and-excitation (03.05).

## ReLU and its variants

$$\mathrm{ReLU}(x) = \max(0, x), \qquad \mathrm{ReLU}'(x) = \mathbb{1}[x > 0]$$

ReLU made deep supervised nets practical (AlexNet, 2012). For active units the gradient passes through unchanged, so there is no per-layer shrinkage; it is also one comparison to compute. Its failure mode is the **dead unit**: if a unit's pre-activation is negative for every input (often after one large update pushes its bias very negative), its gradient is zero for every example and it never recovers. Fixes:

- **Leaky ReLU**: $\max(\alpha x, x)$ with $\alpha \approx 0.01$, so negative inputs still carry a small gradient.
- **ELU**: $\alpha(e^x - 1)$ for $x<0$, smooth and with negative mean outputs.
- Lower learning rate and proper initialisation (01.05) prevent most deaths in the first place.

## GELU and SiLU: smooth gates

Modern Transformers and many CNNs use activations that behave like ReLU for large $|x|$ but are smooth near zero and allow small negative outputs:

$$\mathrm{GELU}(x) = x\,\Phi(x) \approx 0.5x\left(1 + \tanh\!\left(\sqrt{2/\pi}\,(x + 0.044715x^3)\right)\right)$$
$$\mathrm{SiLU}(x) = x\,\sigma(x) \quad \text{(also called Swish)}$$

Read them as "multiply the input by a soft probability that it should pass". Both are non-monotonic: a minimum around $x \approx -0.75$ (GELU) or $x \approx -1.28$ (SiLU). Smoothness gives better-behaved second-order structure and empirically small but consistent accuracy gains in large models. Gated variants such as SwiGLU multiply two projections, one through SiLU; AI Research covers that in its SwiGLU lesson.

| Activation | Range | Max derivative | Main failure |
| --- | --- | --- | --- |
| Sigmoid | (0, 1) | 0.25 | saturation, not zero-centred |
| Tanh | (-1, 1) | 1 | saturation |
| ReLU | [0, ∞) | 1 | dead units |
| Leaky ReLU | (-∞, ∞) | 1 | extra hyperparameter |
| GELU / SiLU | about (-0.17 / -0.28, ∞) | about 1.1 | costlier to compute |

~~~dg-flow How the choice of activation shows up in training: the derivative at each layer multiplies the backward signal.
Upstream grad -(× φ'(z) at layer L)-> Layer L-1 -(× φ'(z))-> Layer L-2 -(sigmoid: ≤ 0.25 each)-> *Vanished at layer 1
~~~

## Numerically safe forms

Naive formulas overflow. `np.exp(-x)` for $x = -1000$ is `inf`; dividing gives 0 with a warning, and anything downstream that takes a log produces `-inf` or `nan`. The fixes all rely on never exponentiating a large positive number:

```python
import numpy as np

def sigmoid(x):
    # e^{-|x|} is always in (0, 1]: no overflow on either side
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))

def softplus(x):            # log(1 + e^x), the smooth ReLU
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

def log_sigmoid(x):         # log σ(x) = -softplus(-x)
    return -softplus(-x)

def tanh(x):
    return np.tanh(x)       # library tanh is already stable; 2σ(2x)-1 loses precision near 0
```

Two related rules carry into the next lesson: compute `log(softmax)` with log-sum-exp rather than taking the log of a softmax, and feed **logits** to loss functions (`BCEWithLogitsLoss`, `CrossEntropyLoss`) instead of probabilities, so the library can use these stable forms.

<div class="ex">
<div class="lab">Worked example</div>

**how much gradient survives 6 layers**

Suppose every pre-activation is near zero, the best case for sigmoid. Each layer multiplies the gradient by $\sigma'(0) = 0.25$, so after 6 layers $0.25^6 \approx 2.4 \times 10^{-4}$ of the signal remains, before counting weights.

With tanh, $\tanh'(0) = 1$, so near zero the signal survives; but if pre-activations drift to $|z| = 3$, $\tanh'(3) \approx 0.0099$ and six layers leave about $10^{-12}$.

With ReLU, active units pass a factor of exactly 1; the risk moves from shrinking gradients to units being off entirely.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

pick ReLU-family or GELU/SiLU for hidden layers, keep sigmoid and tanh for outputs and gates, and always implement activations and their logs in forms that never exponentiate a large positive number.

</div>
