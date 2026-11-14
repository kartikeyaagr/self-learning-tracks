---
id: "01.05"
course: "deep-learning"
module: "01"
slug: "weight-initialisation"
title: "Weight initialisation"
keyIdeas: ["For a layer $z = Wx$ with independent zero-mean entries, $\\mathrm{Var}(z_i) = n_{in}\\,\\mathrm{Var}(w)\\,\\mathrm{Var}(x)$, so the weight variance must scale like $1/n_{in}$ to keep signals the same size layer after layer.", "Xavier/Glorot sets $\\mathrm{Var}(w) = 2/(n_{in}+n_{out})$, a compromise between preserving forward activations and backward gradients, derived for tanh-like activations.", "He/Kaiming sets $\\mathrm{Var}(w) = 2/n_{in}$ for ReLU, doubling the variance because ReLU zeroes half its inputs.", "Initialising all weights to the same value (including zero) makes every hidden unit compute the same thing forever; random init exists to break symmetry."]
question: "You build a 40-layer plain (no residual, no normalisation) tanh network and initialise with He init because \"it's the modern default\". Training stalls immediately and activations in late layers are mostly near ±1. What went wrong, what init would you use instead, and why does the answer differ between tanh and ReLU?"
answer: "He init sets $\\mathrm{Var}(w) = 2/n_{in}$, which doubles the variance to compensate for ReLU discarding half its inputs. Tanh discards nothing; near zero it is approximately the identity, so with He init each layer's pre-activation variance roughly doubles until tanh saturates. Saturated units (outputs near ±1) have derivative $1 - \\tanh^2 \\approx 0$, so gradients vanish backward through 40 layers and training stalls. For tanh the right choice is Xavier/Glorot, $\\mathrm{Var}(w) = 2/(n_{in}+n_{out})$, or LeCun $1/n_{in}$, which keep the per-layer gain about 1 in the near-linear regime. The difference comes from the activation's second moment: ReLU halves $\\mathbb{E}[h^2]$ and needs a gain-2 correction, tanh roughly preserves it for small inputs and needs none. PyTorch encodes this as `nn.init.calculate_gain('relu') = sqrt(2)` versus `5/3` for tanh."
sources: [{"title": "Glorot & Bengio: Understanding the difficulty of training deep feedforward neural networks", "url": "https://proceedings.mlr.press/v9/glorot10a.html", "note": "Origin of Xavier init, with measurements of activation and gradient variance by layer."}, {"title": "He et al.: Delving Deep into Rectifiers", "url": "https://arxiv.org/abs/1502.01852", "note": "Section 2.2 derives the $2/n_{in}$ rule for ReLU and shows a 30-layer net that only trains with it."}, {"title": "PyTorch docs: torch.nn.init", "url": "https://pytorch.org/docs/stable/nn.init.html", "note": "Exact formulas for xavier_*, kaiming_* and calculate_gain."}, {"title": "Dive into Deep Learning: Numerical Stability and Initialization", "url": "https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html", "note": "Vanishing and exploding gradients and the symmetry problem, with code."}]
practice: ["weight-initialization", "multi-layer-perceptron-forward-pass", "diagnose-mup-width-invariants"]
tags: ["initialisation", "variance preservation"]
tools: ["NumPy", "PyTorch"]
---

## Why the starting point matters

Gradient descent needs a starting point, and for deep networks a bad one is fatal before the first update. If each layer multiplies the signal's scale by 0.9, a 50-layer network shrinks it by $0.9^{50} \approx 0.005$; by 1.1, it grows by about 117×. The backward pass multiplies by the same weights transposed, so gradients vanish or explode the same way. Initialisation is about choosing the weight scale so that this per-layer factor is close to 1. Batch and layer normalisation (02.01, 02.02) and residual connections (03.04) make networks far less sensitive to init, but they don't remove the need for it.

## Symmetry breaking

If every weight in a layer starts at the same value, every hidden unit receives the same input, computes the same output, and receives the same gradient. They stay identical forever and the layer behaves like a single unit. Random initialisation breaks this symmetry. Biases have no such problem and are usually set to zero.

## The variance argument

Take one pre-activation $z_i = \sum_{j=1}^{n_{in}} w_{ij} x_j$, with weights and inputs independent, weights zero-mean. Then

$$\mathrm{Var}(z_i) = n_{in}\,\mathrm{Var}(w)\,\mathbb{E}[x_j^2]$$

To keep the scale constant across layers you need $n_{in}\,\mathrm{Var}(w) \approx 1$ (adjusted for the activation). The backward pass sums over $n_{out}$ outputs instead, so preserving gradient variance needs $n_{out}\,\mathrm{Var}(w) \approx 1$. Three classical schemes follow:

| Scheme | Var(w) | Derived for | Default in |
| --- | --- | --- | --- |
| LeCun | $1/n_{in}$ | linear / SELU | SELU networks |
| Xavier (Glorot) | $2/(n_{in}+n_{out})$ | tanh, sigmoid near 0 | `nn.init.xavier_*` |
| He (Kaiming) | $2/n_{in}$ | ReLU | CNNs with ReLU |

**Xavier** averages the forward and backward conditions. **He** adds a factor 2 for ReLU: ReLU sets half its inputs to zero, so $\mathbb{E}[\mathrm{ReLU}(z)^2] = \tfrac{1}{2}\mathrm{Var}(z)$, and the weights must make up the lost half. He et al. showed this choice lets a 30-layer plain ReLU net converge where Xavier init stalls.

For a uniform distribution $U(-a, a)$ the variance is $a^2/3$, so Xavier-uniform uses $a = \sqrt{6/(n_{in}+n_{out})}$ and He-uniform $a = \sqrt{6/n_{in}}$. For convolutions, $n_{in} = C_{in} k^2$ (the fan-in of one output pixel).

~~~dg-flow The per-layer gain decides everything: a factor slightly off 1 compounds over depth.
Input var 1 -(gain g per layer)-> Layer 10: g^10 -(× g^40 more)-> *Layer 50: g^50 -(g = 0.9 → 0.005, g = 1.1 → 117)-> Vanish or explode
~~~

```python
import numpy as np

def activation_stds(init_std, depth=50, width=512, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(1000, width))
    stds = []
    for _ in range(depth):
        W = rng.normal(0, init_std, (width, width))
        h = np.maximum(h @ W, 0)
        stds.append(h.std())
    return stds

for s in (0.01, np.sqrt(2 / 512), 0.1):
    print(f"{s:.4f}", f"{activation_stds(s)[-1]:.3g}")
# 0.0100 -> collapses to ~0, 0.0625 (He) -> stays order 1, 0.1000 -> blows up
```

<div class="ex">
<div class="lab">Worked example</div>

**three inits for a 50-layer ReLU MLP of width 512**

With ReLU each layer multiplies the second moment by about $n\,\sigma^2 / 2 = 256\,\sigma^2$.

$\sigma = 0.01$: gain $0.0256$ per layer, so after 50 layers the variance is about $0.0256^{50} \approx 10^{-80}$: every activation is effectively zero and so is every gradient.

$\sigma = 0.1$: gain $2.56$ per layer, $2.56^{50} \approx 10^{20}$: activations pass fp16's maximum of 65504 after about 20 layers.

$\sigma = \sqrt{2/512} = 0.0625$ (He): gain exactly 1; activations stay order one at every depth.

</div>

## Practical notes

- **Framework defaults differ.** PyTorch's `nn.Linear` and `nn.Conv2d` default to Kaiming-uniform with $a = \sqrt{5}$, which works out to $U(-1/\sqrt{n_{in}}, 1/\sqrt{n_{in}})$, variance $1/(3n_{in})$: smaller than He. It trains fine with normalisation layers but is worth overriding in a deep plain ReLU net.
- **Residual branches.** In a residual network the output is $x + F(x)$, so variance adds up with depth even if each branch is well initialised. Common fixes: zero-init the last BatchNorm scale in each block (each block starts as identity), or scale the residual projections by $1/\sqrt{2N}$ for $N$ layers as GPT-2 does.
- **Output layer.** Initialising logits small (or zero) means the initial loss is near $\ln C$, which is the first sanity check in 02.05.
- **Width scaling.** These rules fix the scale at initialisation only. How learning rates should scale with width (muP, the Maximal Update Parametrisation) is a separate question beyond this course; the practice problem on muP width invariants is a starting point.

<div class="take">
<div class="lab">Key takeaway</div>

choose the weight variance so each layer's gain is about 1, $2/n_{in}$ for ReLU and $2/(n_{in}+n_{out})$ for tanh, and break symmetry with randomness; with normalisation and residuals in place, init matters less but still decides whether step one is sane.

</div>
