---
id: "01.02"
course: "deep-learning"
module: "01"
slug: "activation-functions"
title: "Activation functions"
keyIdeas: ["An activation function $\\phi$ is applied to every unit's weighted sum; without it a stack of layers is one linear map. During backprop, each layer multiplies the gradient by the activation's slope $\\phi'(z)$.", "Sigmoid and tanh saturate: for large $|z|$ their slope goes to zero, and sigmoid's slope never exceeds 0.25, so stacked sigmoids shrink gradients geometrically (vanishing gradients).", "ReLU has slope exactly 1 for positive inputs, which keeps deep gradients alive, but a unit whose input stays negative for every example gets zero gradient forever (a dead unit). Large learning rates cause this; Leaky ReLU, He init and lower rates prevent it.", "GELU, $z\\Phi(z)$, and SiLU, $z\\sigma(z)$, are smooth ReLU-like gates that let small negative values through; they are the defaults in Transformers.", "Use ReLU-family or GELU/SiLU in hidden layers, no activation on regression outputs or logits, and sigmoid only for probabilities and gates.", "Write activations in numerically safe forms: compute sigmoid and softplus with branches or $\\log(1+e^{-|z|})$ so `exp` never overflows, and give logits, not probabilities, to the loss."]
question: "A 20-layer tanh network trains well at learning rate 1e-3 but at 1e-1 the loss stops improving within a few hundred steps, and inspecting activations shows most values at ±1. A ReLU version at 1e-1 stalls too, but its activations are mostly exactly 0. Explain both failures in terms of the activation derivatives, and suggest one fix for each."
answer: "Tanh: the large learning rate pushes weights large, so pre-activations land deep in the flat tails where $\\tanh'(z) = 1 - \\tanh^2(z) \\approx 0$ (already 0.0099 at $|z| = 3$). Every saturated layer multiplies the backward signal by nearly zero, so over 20 layers the gradient reaching the early layers vanishes and learning stalls, even though the units are not dead in principle: if their inputs returned towards zero they would learn again. Diagnose it by measuring the share of outputs with $|h| > 0.99$ and the gradient norm per layer, which falls steadily towards the input. Fixes: a lower learning rate (or warmup), Xavier initialisation, or normalisation layers that keep pre-activations near zero. ReLU: a few big updates drove many units' biases and weights so that their pre-activations are negative for every input. Their derivative is then exactly 0 on every example, so their weights never receive a gradient and they never come back (dead units); with enough of them the downstream layers see almost constant input and the loss sits near chance. The signal is not shrinking smoothly, it is cut off. Diagnose it by counting units whose output is 0 on a whole batch. Fixes: lower learning rate or warmup, He initialisation, or Leaky ReLU/GELU so negative inputs keep a small gradient; dead units do not recover, so restart training after the fix. Both cases show that the activation's derivative profile determines which way a too-large step breaks training."
sources: [{"title": "Hendrycks & Gimpel: Gaussian Error Linear Units (GELUs)", "url": "https://arxiv.org/abs/1606.08415", "note": "Definition of GELU, the tanh approximation and comparisons with ReLU and ELU."}, {"title": "Ramachandran, Zoph, Le: Searching for Activation Functions", "url": "https://arxiv.org/abs/1710.05941", "note": "Where Swish ($x\\sigma(\\beta x)$) came from, found by automated search."}, {"title": "CS231n: Neural Networks Part 1 (activation functions)", "url": "https://cs231n.github.io/neural-networks-1/", "note": "Sigmoid saturation, non-zero-centred outputs and dying ReLUs, concisely."}, {"title": "PyTorch docs: torch.nn.GELU", "url": "https://pytorch.org/docs/stable/generated/torch.nn.GELU.html", "note": "The exact and tanh-approximate forms as implemented."}]
practice: ["activation-functions", "derivatives-of-common-activation-functions", "sigmoid-stable", "continuity-of-activation-functions", "sigmoid-boundedness-via-squeeze-theorem"]
tags: ["activations", "saturation", "numerical stability"]
tools: ["NumPy"]
---

## What you will learn

- What an activation function does inside a neuron, and why a network without one collapses to a straight line.
- The shapes and the slopes of the five activations you will meet most: sigmoid, tanh, ReLU, GELU and SiLU.
- Why the slope of the activation decides whether gradients survive a deep network (vanishing gradients and saturation).
- What a "dead" ReLU unit is, how a too-large learning rate creates many of them, and the variants that avoid it.
- How to write activations so they never overflow, and how to pick one for hidden layers and for outputs.

*Before you start:* you need lesson 01.01 (a neuron computes $\phi(w^\top x + b)$, and the nonlinearity $\phi$ is what makes depth meaningful). You need the idea of a derivative and the chain rule: backprop multiplies one derivative per layer. The full backward pass is in 01.04; here we only need that multiplication.

## From "we need a nonlinearity" to "which one?"

Lesson 01.01 showed that stacking linear layers gives you just another linear layer, so every neuron passes its weighted sum through a nonlinear function $\phi$. That function is the **activation function**. The previous lesson treated it as a black box. This lesson opens the box, because the choice of $\phi$ decides two practical things:

- whether the gradient signal reaches the early layers during training, or fades to nothing on the way;
- whether the forward computation stays finite in floating point.

## The idea in one picture

Take a single neuron and feed it five different weighted sums. The weighted sum $z = w^\top x + b$ is called the **pre-activation**. The activation turns each $z$ into an output $h = \phi(z)$:

| Pre-activation z | -3 | -1 | 0 | 1 | 3
| --- | --- | --- | --- | --- | --- |
| sigmoid | 0.047 | 0.269 | 0.500 | 0.731 | 0.953
| tanh | -0.995 | -0.762 | 0.000 | 0.762 | 0.995
| ReLU | 0 | 0 | 0 | 1 | 3
| GELU | -0.004 | -0.159 | 0 | 0.841 | 2.996
| SiLU | -0.142 | -0.269 | 0 | 0.731 | 2.858

Sigmoid squeezes everything into (0, 1) and tanh into (-1, 1); at $z = 3$ both are nearly at their ceiling. ReLU simply deletes negatives and keeps positives unchanged. GELU and SiLU look like ReLU at the ends but bend smoothly near zero and let small negative values through.

Why does any of this matter? Fit the same small network (two hidden layers of 32 units) to a wavy curve, changing only $\phi$:

~~~dg-fig 01.02-fit.svg
Same architecture, same optimiser, three activations. With no activation the network can only draw a straight line (training MSE 0.436); with ReLU or tanh it follows the wave (MSE about 0.002).
~~~

With $\phi$ = identity, two layers of 32 units are mathematically one linear map, so the best it can do is a tilted line. Either nonlinearity lets it bend. ReLU builds the curve from straight segments joined at kinks; tanh builds it from smooth S-shapes. At this size the choice barely matters; in deep networks it matters a lot.

## Step 1: what an activation does, and the number backprop cares about

An activation is applied **elementwise**: every unit in a layer gets its own $z$ and applies the same $\phi$ to it. For a whole layer,

$$h = \phi(Wx + b),$$

where $W$ is the weight matrix, $b$ the bias vector, $Wx + b$ the vector of pre-activations, and $h$ the vector of outputs (the **activations**).

In training, backprop sends a gradient backward through this layer. By the chain rule, the gradient at $z$ is the gradient at $h$ times the activation's slope:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h}\cdot \phi'(z).$$

Here $L$ is the loss and $\phi'(z)$ is the derivative of the activation evaluated at this unit's pre-activation. So every layer multiplies the backward signal by $\phi'(z)$ (and by its weights). A tiny factor blocks learning for everything below; a zero lets nothing pass.

~~~dg-fig 01.02-curves.svg
Left: the five activations. Right: their slopes. Sigmoid's slope never exceeds 0.25; sigmoid and tanh go flat (slope near 0) once |z| is beyond about 3; ReLU's slope is exactly 1 for positive inputs and exactly 0 for negative ones.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one neuron, by hand**

Input $x = (2, -1)$, weights $w = (0.5, 1.5)$, bias $b = 0.2$. Pre-activation: $z = 0.5 \times 2 + 1.5 \times (-1) + 0.2 = 1 - 1.5 + 0.2 = -0.3$.

Outputs: ReLU gives $\max(0, -0.3) = 0$. Sigmoid gives $1/(1 + e^{0.3}) = 1/(1 + 1.350) = 0.426$. Tanh gives $-0.291$. SiLU gives $-0.3 \times 0.426 = -0.128$. GELU gives $-0.3 \times \Phi(-0.3) = -0.3 \times 0.382 = -0.115$, where $\Phi$ is the standard normal cumulative probability.

Slopes: sigmoid $\sigma(z)(1 - \sigma(z)) = 0.426 \times 0.574 = 0.244$. Tanh $1 - \tanh^2 z = 1 - 0.085 = 0.915$. ReLU: $z < 0$, so the slope is 0.

Consequence: if the gradient arriving at $h$ is 1, the ReLU neuron passes 0 back, so its weights get no update from this example. The sigmoid neuron passes 0.244 and the tanh neuron 0.915. One example with a zero gradient is normal; a unit that gets zero for every example is the "dead unit" problem of Step 3.

</div>

## Step 2: sigmoid and tanh, smooth but saturating

The classic pair:

$$\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = \sigma(z)\big(1-\sigma(z)\big) \le 0.25,$$

$$\tanh(z) = 2\sigma(2z) - 1, \qquad \tanh'(z) = 1 - \tanh^2(z) \le 1.$$

Both squash their input into a bounded range: (0, 1) for **sigmoid** and (-1, 1) for **tanh**. Their weakness is **saturation**: when $|z|$ is large, the curve is flat, the derivative is close to zero, and backprop multiplies the gradient by that near-zero number. From the table, at $z = 3$ the sigmoid slope is 0.045 and the tanh slope is 0.010.

Even in the best case, sigmoid's slope is 0.25 (at $z = 0$). Stack ten sigmoid layers and the activation factors alone shrink the gradient by at least $0.25^{10} \approx 10^{-6}$. This is the **vanishing gradient** problem. We measured it: one backward pass through a 20-hidden-layer network (64 units each, standard initialisation) on a batch of handwritten digits:

~~~dg-fig 01.02-vanish.svg
Gradient size per layer at initialisation. With sigmoid the first layer's gradient is about 10¹¹ times smaller than the last layer's, so the early layers effectively never learn. Tanh, ReLU and GELU keep it roughly level.
~~~

The sigmoid line falls on average by a factor of about 4 per layer, just as a slope of at most $0.25$ per layer predicts: $0.25^{19} \approx 4 \times 10^{-12}$, and the measured ratio between layer 1 and layer 20 is $8.6 \times 10^{-12}$. Tanh survives here because its slope at zero is 1 and suitable initialisation (01.05) keeps pre-activations near zero.

Sigmoid has a second, subtler flaw: it is not **zero-centred**. Its outputs are all positive, so for one example the gradients on all the weights feeding the next neuron share one sign. They can all go up together or all go down together, but not some of each, so the updates zig-zag towards the target. Tanh is zero-centred and fixes this, but it still saturates.

<div class="ex">
<div class="lab">Worked example</div>

**how much gradient survives 6 layers**

Suppose every pre-activation is near zero, the best case for sigmoid. Each layer multiplies the gradient by $\sigma'(0) = 0.25$, so after 6 layers $0.25^6 = 0.000244 \approx 2.4 \times 10^{-4}$ of the signal remains, before counting weights.

With tanh, $\tanh'(0) = 1$, so near zero the signal survives. But if the pre-activations drift to $|z| = 3$, then $\tanh'(3) = 1 - 0.99505^2 \approx 0.0099$, and six layers leave $0.0099^6 \approx 9 \times 10^{-13}$, about $10^{-12}$.

With ReLU, active units pass a factor of exactly 1. The risk moves from gradients that shrink to units that are switched off entirely.

</div>

Sigmoid is still the right tool where you want a number in (0, 1): a binary classifier's output, LSTM gates, gating blocks such as squeeze-and-excitation (03.05). It is a poor choice for deep hidden layers.

## Step 3: ReLU, and the dead unit

The **rectified linear unit** is

$$\mathrm{ReLU}(z) = \max(0, z), \qquad \mathrm{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0. \end{cases}$$

At exactly $z = 0$ the derivative is undefined; libraries simply pick a value (PyTorch uses 0). ReLU made deep supervised networks practical (AlexNet, 2012, is the famous example). For active units the gradient passes through unchanged, so there is no per-layer shrinkage, and it costs one comparison to compute. Its outputs are also **sparse**: many units are exactly 0 for a given input.

Its failure mode is the **dead unit**. If a unit's pre-activation is negative for *every* input in the data, its output is always 0, its slope is always 0, and so its incoming weights never receive a gradient. Nothing can bring it back. Typically one large update pushed its bias strongly negative. We provoked it with a 4-hidden-layer ReLU network on the digits data:

~~~dg-fig 01.02-dead.svg
Left: share of the 256 hidden units that output 0 for every image. With learning rate 0.5, about 60% die within the first 50 steps and the loss sticks at chance (2.30). With learning rate 0.05, essentially none die (0.4% at the end) and the loss goes to almost zero.
~~~

Fixes, from most to least common:

- **Lower learning rate, warmup and proper initialisation** (He initialisation, 01.05). These prevent most deaths in the first place.
- **Leaky ReLU**: $\max(\alpha z, z)$ with $\alpha \approx 0.01$. Negative inputs keep a small slope, so a unit that drifts negative still gets a gradient and can recover.
- **ELU**: $z$ for $z \ge 0$ and $\alpha(e^z - 1)$ for $z < 0$. Smooth, saturates gently at $-\alpha$ on the negative side, and its outputs average closer to zero.

## Step 4: GELU and SiLU, smooth gates

Modern Transformers and many recent CNNs use activations that behave like ReLU for large $|z|$ but are smooth near zero and allow small negative outputs:

$$\mathrm{GELU}(z) = z\,\Phi(z) \approx 0.5z\left(1 + \tanh\!\left(\sqrt{2/\pi}\,(z + 0.044715z^3)\right)\right),$$

$$\mathrm{SiLU}(z) = z\,\sigma(z) \quad \text{(also called Swish).}$$

Here $\Phi(z)$ is the probability that a standard normal variable is below $z$. Read both as "multiply the input by a soft probability that it should pass". For large positive $z$ that probability is about 1 (output $\approx z$, like ReLU); for large negative $z$ it is about 0 (output $\approx 0$); in between, the gate is partly open.

Both are **non-monotonic**: they dip below zero before coming back. GELU's minimum is about $-0.17$ at $z \approx -0.75$; SiLU's is about $-0.28$ at $z \approx -1.28$. Their slopes slightly overshoot 1 (maximum about 1.13 for GELU, 1.10 for SiLU). In large models they give small but consistent accuracy gains over ReLU, so they are the defaults in Transformers. Gated variants such as SwiGLU multiply two projections, one passed through SiLU; AI Research covers that in its SwiGLU lesson.

| Activation | Range | Max slope | Zero-centred | Main failure
| --- | --- | --- | --- | --- |
| Sigmoid | (0, 1) | 0.25 | no | saturation, vanishing gradients
| Tanh | (-1, 1) | 1 | yes | saturation
| ReLU | [0, ∞) | 1 | no | dead units
| Leaky ReLU | (-∞, ∞) | 1 | roughly | one extra hyperparameter
| GELU / SiLU | about [-0.17, ∞) / [-0.28, ∞) | about 1.13 / 1.10 | roughly | costlier to compute

Putting it all together, here is a 10-hidden-layer network trained on the digits with the same optimiser (SGD with momentum, learning rate 0.05) and four activations:

~~~dg-fig 01.02-train.svg
Same network, same data, same optimiser. Sigmoid never leaves chance level (loss 2.32 after 300 steps); tanh and GELU reach about 0.002; ReLU trains too (0.10 at the end) but shows loss spikes at this learning rate.
~~~

~~~dg-flow How the choice of activation shows up in training: the slope at each layer multiplies the backward signal.
Upstream grad -(× φ'(z) at layer L)-> Layer L-1 -(× φ'(z))-> Layer L-2 -(sigmoid: ≤ 0.25 each)-> *Vanished at layer 1
~~~

## Activations in code

In PyTorch every activation is a module (and also a function in `torch.nn.functional`). Autograd computes the slopes for you; this prints outputs and slopes at seven points:

```python
import torch, torch.nn as nn

z = torch.linspace(-3, 3, 7, requires_grad=True)          # -3, -2, ..., 3
for act in [nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.LeakyReLU(0.01), nn.GELU(), nn.SiLU()]:
    out = act(z)
    slope, = torch.autograd.grad(out.sum(), z)
    print(type(act).__name__, [round(v, 3) for v in out.tolist()], [round(v, 3) for v in slope.tolist()])
# Sigmoid  [0.047, 0.119, 0.269, 0.5, 0.731, 0.881, 0.953]  [0.045, 0.105, 0.197, 0.25, 0.197, 0.105, 0.045]
# Tanh     [-0.995, -0.964, -0.762, 0.0, 0.762, 0.964, 0.995] [0.01, 0.071, 0.42, 1.0, 0.42, 0.071, 0.01]
# ReLU     [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]  [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]   <- slope 0 at z = 0
# LeakyReLU [-0.03, -0.02, -0.01, 0.0, 1.0, 2.0, 3.0] [0.01, 0.01, 0.01, 0.01, 1.0, 1.0, 1.0]
# GELU     [-0.004, -0.046, -0.159, 0.0, 0.841, 1.954, 2.996] [-0.012, -0.085, -0.083, 0.5, 1.083, 1.085, 1.012]
# SiLU     [-0.142, -0.238, -0.269, 0.0, 0.731, 1.762, 2.858] [-0.088, -0.091, 0.072, 0.5, 0.928, 1.091, 1.088]

print(nn.GELU()(torch.tensor([1.0])).item(), nn.GELU(approximate='tanh')(torch.tensor([1.0])).item())
# 0.8413  0.8412   (exact erf form vs the tanh approximation; they differ by < 0.0005 everywhere)
```

## Numerically safe forms

If you implement activations yourself, the naive formulas can overflow. `np.exp(-z)` for $z = -1000$ is `inf`; the sigmoid then comes out as 0 with a warning, and anything that later takes a log gets `-inf` or `nan`. The fixes all follow one rule: never exponentiate a large positive number.

```python
import numpy as np

def sigmoid(z):
    e = np.exp(-np.abs(z))                  # always in (0, 1]: cannot overflow
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

def softplus(z):                            # log(1 + e^z), the smooth ReLU
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))

def log_sigmoid(z):                         # log σ(z) = -softplus(-z)
    return -softplus(-z)

z = np.array([-1000., -20., 0., 20., 1000.])
print(np.log(1 / (1 + np.exp(-z))))   # [-inf, -20, -0.693, -2.06e-09, 0]   + overflow warning
print(log_sigmoid(z))                 # [-1000, -20, -0.693, -2.06e-09, -0]  correct everywhere
```

Use the library's `np.tanh` rather than $2\sigma(2z) - 1$, which loses precision near zero. The same rule carries into 01.03: feed **logits** (raw pre-activations) to `BCEWithLogitsLoss` or `CrossEntropyLoss` instead of probabilities, so the library can use stable forms.

## Choosing an activation and checking it works

A short decision list covers almost every case:

- **Hidden layers of an MLP or CNN:** ReLU is a fine default. Try GELU or SiLU if you are matching a modern architecture or chasing the last bit of accuracy.
- **Transformers:** GELU, or SwiGLU in the gated feed-forward block, as the architecture you copy specifies.
- **Output layer:** no activation for regression or for logits fed to a loss; sigmoid only for a probability or a gate.

Whatever you choose, the initialisation must match it (01.05): Xavier/Glorot for tanh and sigmoid, He/Kaiming for ReLU-family activations.

To check activations in a real network, log per-layer statistics (a forward hook works) after a few hundred steps:

- **Dead fraction (ReLU):** the share of units whose output is 0 on every example in a large batch. A few percent is normal; above about 10-20% and rising means trouble.
- **Saturated fraction (sigmoid/tanh):** the share of outputs with $|h| > 0.99$ (tanh) or outside (0.01, 0.99) (sigmoid). Many saturated units mean small gradients.
- **Gradient norm per layer:** as in the vanishing figure; a steady fall towards the input means vanishing gradients.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss sits at chance from step 1 in a deep net | sigmoid (or saturated tanh) hidden layers, vanishing gradients | ReLU/GELU in hidden layers; Xavier/He init; normalisation layers (02.01)
| Loss drops, then flat; many activations exactly 0 | dead ReLUs after a large update | lower learning rate or warmup; Leaky ReLU, GELU; He init
| Most tanh outputs at ±1 | weights grew too large, saturation | lower learning rate, Xavier init, normalisation
| `nan` or `inf` in the loss | naive `exp` overflowed in a hand-written sigmoid or softmax | stable forms above; give logits to the loss
| Output layer can't predict negative or large values | ReLU or sigmoid left on a regression output | no activation on the output

## Going deeper

**Why "zero-centred" matters.** For a neuron with $z = \sum_j w_j h_j + b$, the gradient on weight $w_j$ is $\frac{\partial L}{\partial z} h_j$. If every input $h_j$ is positive (sigmoid or ReLU outputs), all those gradients have the sign of $\frac{\partial L}{\partial z}$, so for one example the weight vector can only move in directions where all components rise or all fall. Mixed moves then take a zig-zag of steps; mini-batch averaging and normalisation layers soften this.

**ReLU networks are piecewise linear.** Each ReLU unit splits input space with a hyperplane (where $z = 0$). Within any region where the on/off pattern of all units is fixed, the network is exactly an affine function. That is why the ReLU fit in the first figure is made of straight segments. The number of such linear regions can grow exponentially with depth, which is one formal reason depth helps (01.01).

**Where GELU comes from.** Imagine multiplying each input by a random mask $m \sim \text{Bernoulli}(\Phi(z))$: keep $z$ with probability $\Phi(z)$, which is larger for larger inputs. The expected output is $z\,\Phi(z)$, which is GELU. So GELU is the deterministic average of an input-dependent dropout. SiLU is the same idea with a logistic instead of a normal gate, and it was also found by automated search over activation formulas (Swish).

**ReLU and initialisation.** A ReLU zeroes about half of a symmetric input distribution, which halves the variance of the signal at each layer. He initialisation compensates with weight variance $2/n_{\text{in}}$ instead of $1/n_{\text{in}}$; that factor of 2 is the gain you will see in 01.05.

**Cost.** ReLU is one comparison; GELU needs an `erf` or `tanh` and SiLU an `exp`. Activations are usually memory-bound and fused into the neighbouring matrix multiply, so the wall-clock difference is small.

## Key terms

| Term | Meaning
| --- | --- |
| Activation function | The nonlinear function $\phi$ each unit applies to its weighted sum
| Pre-activation | The weighted sum $z = w^\top x + b$ before the activation is applied
| Saturation | The flat region of sigmoid or tanh where the slope is close to zero
| Vanishing gradient | Gradients shrinking layer by layer as they flow back to early layers
| Zero-centred | Outputs that average around zero rather than all being positive
| Dead unit | A ReLU unit whose input is negative for every example, so it never gets a gradient
| Leaky ReLU | ReLU with a small slope $\alpha$ for negative inputs, so units cannot die
| GELU / SiLU | Smooth gates $z\Phi(z)$ and $z\sigma(z)$; the defaults in Transformers

<div class="take">
<div class="lab">Key takeaway</div>

the activation's slope multiplies every gradient passing back through a layer, so use ReLU-family or GELU/SiLU in hidden layers, keep sigmoid and tanh for outputs and gates, watch for saturation and dead units, and always implement activations and their logs in forms that never exponentiate a large positive number.

</div>
