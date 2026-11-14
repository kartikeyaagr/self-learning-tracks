---
id: "01.05"
course: "deep-learning"
module: "01"
slug: "weight-initialisation"
title: "Weight initialisation"
keyIdeas: ["If all weights start equal, every hidden unit computes and learns the same thing forever, so weights must be random with mean zero (**symmetry breaking**). Biases can start at zero.", "For $z = \\sum_j w_j x_j$ with independent zero-mean weights, $\\mathrm{Var}(z) = n_{in}\\,\\mathrm{Var}(w)\\,\\mathbb{E}[x^2]$. To keep the signal's size across layers, $n_{in}\\,\\mathrm{Var}(w)$ must be about 1, corrected for the activation.", "**He (Kaiming)** init, $\\mathrm{Var}(w) = 2/n_{in}$, is for ReLU: ReLU zeroes half its inputs and halves the second moment, so the weights double to make up for it.", "**Xavier (Glorot)** init, $\\mathrm{Var}(w) = 2/(n_{in}+n_{out})$, averages the forward ($1/n_{in}$) and backward ($1/n_{out}$) conditions and suits tanh and sigmoid. Using He on tanh overshoots; using $\\sigma = 1$ saturates it.", "A per-layer gain slightly off 1 compounds: with σ = 0.01 or 0.1 a 50-layer ReLU net's activations reach 1e-40 or 1e10. Check activation spread, dead or saturated units, per-layer gradient norms and a first loss near $\\ln C$ before training."]
question: "A colleague builds a 40-layer plain tanh MLP (width 256, no normalisation, no residuals) and creates each weight matrix with `torch.randn(256, 256)`. Training produces NaN losses within a few steps, and a histogram shows about 90% of late-layer activations beyond ±0.95. They propose switching to He initialisation because \"it's the modern default\". What went wrong, is He the right fix, and what would you use instead? Explain why the answer differs between tanh and ReLU."
answer: "`torch.randn` gives $\\sigma_w = 1$, so each layer multiplies the pre-activation variance by about $n_{in}\\,\\mathrm{Var}(w) = 256$. Tanh caps the outputs at ±1, so instead of growing without bound the units saturate: most sit in the flat region where $1 - \\tanh^2(z) \\approx 0$. The network is then in the chaotic regime: the few units that are not saturated pass gradients through very large weights, so gradient sizes vary wildly across layers (in a 40-layer, width-256 test the first layer's gradient norm was about $10^{16}$ against 0.01 at the last layer), which is where the NaNs come from. He initialisation, $\\mathrm{Var}(w) = 2/n_{in}$ ($\\sigma_w = 0.088$ here), is far better than $\\sigma_w = 1$, but it is derived for ReLU. Its factor 2 compensates for ReLU discarding half its inputs, which halves $\\mathbb{E}[h^2]$. Tanh discards nothing: near zero it is approximately the identity, so with He each layer's gain is above 1 and the network stays mildly chaotic (gradients grow toward the input, about 3.5 times over 40 layers in the same test). For tanh the right choice is Xavier/Glorot, $\\mathrm{Var}(w) = 2/(n_{in}+n_{out}) = 1/256$ ($\\sigma_w = 0.0625$), or LeCun $1/n_{in}$, optionally times PyTorch's tanh gain of 5/3. In code: `nn.init.xavier_normal_(layer.weight)` (or `xavier_normal_(w, gain=nn.init.calculate_gain('tanh'))`) and zero biases. Then verify with one batch: activation standard deviation should stay roughly constant across layers with few units beyond ±0.95, per-layer gradient norms should be within an order of magnitude of each other, and the first loss should be near $\\ln C$. For a network this deep, adding normalisation or residual connections would also make it far less sensitive to the exact scale."
sources: [{"title": "Glorot & Bengio: Understanding the difficulty of training deep feedforward neural networks", "url": "https://proceedings.mlr.press/v9/glorot10a.html", "note": "Origin of Xavier init, with measurements of activation and gradient variance by layer."}, {"title": "He et al.: Delving Deep into Rectifiers", "url": "https://arxiv.org/abs/1502.01852", "note": "Section 2.2 derives the $2/n_{in}$ rule for ReLU and shows a 30-layer net that only trains with it."}, {"title": "PyTorch docs: torch.nn.init", "url": "https://pytorch.org/docs/stable/nn.init.html", "note": "Exact formulas for xavier_*, kaiming_* and calculate_gain."}, {"title": "Dive into Deep Learning: Numerical Stability and Initialization", "url": "https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html", "note": "Vanishing and exploding gradients and the symmetry problem, with code."}]
practice: ["weight-initialization", "multi-layer-perceptron-forward-pass", "diagnose-mup-width-invariants"]
tags: ["initialisation", "variance preservation"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why the starting weights must be random ("breaking symmetry").
- How the scale of the starting weights decides whether signals vanish, explode or stay healthy with depth.
- The one-line variance rule behind LeCun, Xavier (Glorot) and He (Kaiming) initialisation.
- How to set and check initialisation in PyTorch, including what its defaults actually do.
- How to spot a bad initialisation from activation and gradient statistics before training.

*Before you start:* you need the MLP forward pass (01.01), activation functions and why ReLU and tanh behave differently (01.02), and backpropagation as repeated multiplication by weight matrices (01.04). From probability: variance measures spread, and variances of independent terms add.

## Where we are

Lesson 01.04 showed that the backward pass multiplies a gradient by one weight matrix per layer, and the forward pass does the same to the signal. Before the first update those matrices are whatever you chose, so your choice decides whether signal and gradient reach the far end of the network at a usable size. Get it wrong and training is dead before it starts.

## The idea in one picture

Imagine each layer multiplies the typical size of the signal by some factor $g$, the **per-layer gain**. Over $L$ layers the factors multiply:

| Gain per layer | After 10 layers | After 50 layers
| --- | --- | --- |
| 0.9 | 0.35 | 0.005
| 1.0 | 1 | 1
| 1.1 | 2.6 | 117

A gain 10% away from 1 becomes a factor of about 200 shrinkage or 100 growth by layer 50. Now a real network: 50 ReLU layers of width 512, fed 1000 random inputs, weights drawn from a normal distribution with standard deviation $\sigma$. Only $\sigma$ differs between the lines.

~~~dg-fig 01.05-depth.svg
The spread of the activations, layer by layer, for three weight scales (log scale). σ = 0.01 shrinks the signal to about 1e-40 by layer 50, σ = 0.1 grows it past the fp16 limit around layer 21, and He's σ = 0.0625 keeps it near 1 the whole way.
~~~

With $\sigma = 0.01$ the activations are around $10^{-8}$ by layer 10: the output no longer depends on the input. With $\sigma = 0.1$ they are around 80 by layer 10, and the largest one passes 65504, the biggest number half precision (fp16) can store, at layer 21. The middle value, $\sigma = \sqrt{2/512} \approx 0.0625$, keeps the spread between about 0.5 and 1 at every depth.

That middle value comes from a two-line variance calculation, built step by step below. Two words first:

- **Weight initialisation** is the rule for drawing the starting weights (and biases) before training.
- **Fan-in** $n_{in}$ is the number of inputs feeding one unit; **fan-out** $n_{out}$ is the number of units a layer's output feeds. For a dense 784 → 256 layer, $n_{in} = 784$ and $n_{out} = 256$.

## Step 1: the weights must be random (symmetry breaking)

Why not start every weight at the same value, say 0 or 0.5? If all hidden units in a layer have identical incoming weights, they compute identical outputs. Backpropagation then hands each of them an identical gradient, so after the update they are still identical. The layer has 8 or 800 units on paper but behaves like one unit copied many times. All-zero is the extreme case: ReLU or tanh hidden units then output zero, and most weights get no gradient at all.

~~~dg-fig 01.05-symmetry.svg
A tanh MLP with 8 hidden units on the two-moons data. Started with every weight at 0.5 (blue), it stalls at loss 0.28 and its 8 units never become different; started with random Xavier weights (accent), it reaches loss 0.06.
~~~

The right panel tracks how different the 8 hidden units' weight vectors are. With constant initialisation that spread is exactly 0 for all 1500 steps, so the network can only draw the boundary of a single tanh unit: 88.7% training accuracy, against 98.3% for the random start.

So weights must be random with mean zero. **Biases** do not need this (random weights already make units differ) and usually start at 0. The open question is the spread of the random numbers.

## Step 2: how one layer changes the variance

Take one **pre-activation**, the weighted sum a unit computes before its activation function:

$$z_i = \sum_{j=1}^{n_{in}} w_{ij}\, x_j.$$

Here $x_j$ are the inputs, $w_{ij}$ the weight from input $j$ to unit $i$. Assume weights are independent of each other and of the inputs, with mean zero. Each term $w_{ij} x_j$ then has mean zero and variance $\mathrm{Var}(w)\,\mathbb{E}[x^2]$, and the $n_{in}$ variances add:

$$\mathrm{Var}(z_i) = n_{in}\,\mathrm{Var}(w)\,\mathbb{E}[x^2].$$

In words: output spread = (number of inputs) × (weight spread) × (typical squared input). Notice it is $\mathbb{E}[x^2]$, the **second moment** of the input, not its variance. The two agree when inputs have mean zero, as standardised data does, but ReLU outputs are never negative, so for later layers the distinction matters.

To keep the spread constant, $n_{in}\,\mathrm{Var}(w)$ must be about 1 (corrected for the activation in Step 3). That is **LeCun initialisation**:

$$\mathrm{Var}(w) = \frac{1}{n_{in}}, \qquad \sigma_w = \frac{1}{\sqrt{n_{in}}}.$$

Number check: with 512 standardised inputs and $\sigma_w = 0.01$, $\mathrm{Var}(z) = 512 \times 0.0001 \times 1 = 0.0512$, a 20-fold drop in one layer. With $\sigma_w = 1/\sqrt{512} = 0.0442$, $\mathrm{Var}(z) = 512 \times 0.00195 \times 1 = 1$.

The classic bug of creating weights with `torch.randn(n_in, n_out)` means $\sigma_w = 1$: every layer multiplies the variance by $n_{in}$, here 512.

## Step 3: put the activation back in (He initialisation)

The next layer sees $h = \phi(z)$, where $\phi$ is the activation, so what must be preserved is $\mathbb{E}[h^2]$.

For **ReLU**, $h = \max(0, z)$. If $z$ is symmetric around zero, half the values become 0 and half pass unchanged, so half the second moment survives:

$$\mathbb{E}[h^2] = \tfrac{1}{2}\,\mathrm{Var}(z).$$

Combining with Step 2, each ReLU layer multiplies the second moment by $\tfrac{1}{2}\, n_{in}\, \mathrm{Var}(w)$. Setting that to 1 gives **He (Kaiming) initialisation**, from He et al. (2015):

$$\mathrm{Var}(w) = \frac{2}{n_{in}}, \qquad \sigma_w = \sqrt{\frac{2}{n_{in}}}.$$

The factor 2 replaces the half that ReLU throws away. For width 512, $\sigma_w = \sqrt{2/512} = 0.0625$, the healthy line in the first figure.

<div class="ex">
<div class="lab">Worked example</div>

**three inits for a 50-layer ReLU MLP of width 512**

Per layer, the second moment is multiplied by $\tfrac{1}{2} \times 512 \times \sigma^2 = 256\,\sigma^2$.

$\sigma = 0.01$: gain $256 \times 0.0001 = 0.0256$. After 50 layers the second moment is $0.0256^{50} \approx 2.6 \times 10^{-80}$, so the typical activation is about $\sqrt{10^{-80}} = 10^{-40}$. The simulation in the figure gives $9 \times 10^{-41}$.

$\sigma = 0.1$: gain $256 \times 0.01 = 2.56$. After 50 layers, $2.56^{50} \approx 2.6 \times 10^{20}$, a typical activation of about $1.6 \times 10^{10}$. The simulation gives $9 \times 10^{9}$, and the largest single activation passes fp16's maximum of 65504 at layer 21.

$\sigma = \sqrt{2/512} = 0.0625$: gain $256 \times 0.0039 = 1$ exactly. Activations stay of order one (0.83 at layer 1, 0.56 at layer 50; finite width adds some drift).

</div>

For **tanh**, nothing is thrown away near zero ($\tanh(z) \approx z$ for small $z$), so the plain rule $\mathrm{Var}(w) \approx 1/n_{in}$ applies. He's factor 2 overshoots on tanh; $\sigma_w = 1$ is far worse. Here is a 20-layer tanh network of width 256:

~~~dg-fig 01.05-tanh.svg
Histograms of tanh activations at layers 1, 10 and 20. With σ = 0.01 they collapse into a spike at 0; with Xavier they stay spread out (slowly narrowing); with σ = 1 about 91% of them sit beyond ±0.95, where tanh is flat.
~~~

The right panel shows **saturation**: a unit pinned near ±1 has derivative $1 - \tanh^2(z) \approx 0$, so it barely responds to input or gradient. The left panel is the opposite: the signal fades to nothing.

## Step 4: the backward pass, and Xavier's compromise

Backpropagation (01.04) sends the gradient back through the same weights, transposed, summing over the $n_{out}$ units that $x_j$ feeds:

$$\frac{\partial L}{\partial x_j} = \sum_{i=1}^{n_{out}} w_{ij}\, \frac{\partial L}{\partial z_i}.$$

The same variance argument now gives a factor $n_{out}\,\mathrm{Var}(w)$ per layer. Keeping gradients the same size therefore wants $\mathrm{Var}(w) = 1/n_{out}$, while the forward pass wanted $1/n_{in}$. When a layer is square ($n_{in} = n_{out}$) the two agree. When it is not, Glorot and Bengio (2010) proposed splitting the difference, which is **Xavier (Glorot) initialisation**:

$$\mathrm{Var}(w) = \frac{2}{n_{in} + n_{out}}.$$

It was derived for tanh and sigmoid in their near-linear region. He et al. showed that preserving either direction alone is enough, because the forward/backward mismatch is a fixed constant that does not compound with depth; He initialisation keeps only the fan-in version.

The next figure runs one forward and one backward pass through a 30-layer ReLU network (width 256) at initialisation and records each layer's weight-gradient size.

~~~dg-fig 01.05-grads.svg
Gradient size for every layer's weights at initialisation. He is roughly flat at 0.2 to 0.5; Xavier is flat but about 10,000 times smaller; PyTorch's default is a billion times smaller near the input than near the output; N(0, 0.01²) gives gradients that are exactly zero in float32.
~~~

- **He** keeps gradients similar in every layer (0.16 at layer 1, 0.55 at layer 30), so every layer learns from step one.
- **Xavier with ReLU** is flat but tiny (about $7 \times 10^{-6}$). Its variance is half He's, so activations halve per layer going forward and gradients halve going backward; a weight gradient is one times the other, so it is uniformly small.
- **$N(0, 0.01^2)$** underflows: the numbers drop below float32's range and are stored as exactly 0. No optimiser recovers from that.

## The three schemes side by side

| Scheme | Var(w) | Derived for | In PyTorch
| --- | --- | --- | --- |
| LeCun | $1/n_{in}$ | linear units, SELU | `kaiming_normal_(w, nonlinearity="linear")`
| Xavier (Glorot) | $2/(n_{in}+n_{out})$ | tanh, sigmoid near 0 | `nn.init.xavier_normal_` / `xavier_uniform_`
| He (Kaiming) | $2/n_{in}$ | ReLU and relatives | `nn.init.kaiming_normal_(w, nonlinearity="relu")`

**Normal or uniform?** Only the variance matters. $U(-a, a)$ has variance $a^2/3$, so for target variance $v$ use $a = \sqrt{3v}$. Xavier-uniform uses $a = \sqrt{6/(n_{in}+n_{out})}$ and He-uniform $a = \sqrt{6/n_{in}}$.

**Convolutions.** One output pixel sees $C_{in}$ channels times a $k \times k$ window, so $n_{in} = C_{in} k^2$: a 3×3 convolution on 64 channels has $n_{in} = 576$ and He standard deviation $\sqrt{2/576} = 0.059$.

**Gain.** PyTorch writes each scheme as $\sigma_w = \text{gain} / \sqrt{\text{fan}}$; `nn.init.calculate_gain` gives $\sqrt{2}$ for ReLU (He's factor 2), 1 for linear and sigmoid, and $5/3$ for tanh, slightly above 1 to counter tanh's mild shrinking.

<div class="ex">
<div class="lab">Worked example</div>

**initialising a 784 → 256 → 256 → 10 ReLU classifier by hand**

Layer 1 (fan-in 784), He normal: $\mathrm{Var}(w) = 2/784 = 0.00255$, so $\sigma_w = \sqrt{0.00255} = 0.0505$. If you prefer uniform: $a = \sqrt{6/784} = 0.0875$.

Check the forward pass: with standardised pixels $\mathbb{E}[x^2] = 1$, so $\mathrm{Var}(z) = 784 \times 0.00255 \times 1 = 2$, and after ReLU $\mathbb{E}[h^2] = 2/2 = 1$. The signal leaves layer 1 at the same scale it entered.

Layer 2 (fan-in 256), He normal: $\sigma_w = \sqrt{2/256} = 0.0884$; uniform bound $\sqrt{6/256} = 0.153$.

Output layer (256 → 10, no ReLU after it): the logits feed softmax, not another ReLU, so use Xavier or even smaller. Xavier: $\mathrm{Var}(w) = 2/266 = 0.0075$, $\sigma_w = 0.087$. Many practitioners shrink it further so the initial logits are near zero and the first loss is close to $\ln 10 = 2.30$ (01.03).

PyTorch's default for layer 1 is $U(-1/\sqrt{784}, 1/\sqrt{784}) = U(-0.0357, 0.0357)$, variance $1/(3 \times 784) = 0.000425$: 6 times smaller than He's 0.00255.

</div>

## Initialisation in code

PyTorch gives every `nn.Linear` and `nn.Conv2d` its own default, so you override it by walking over the modules after building the model. The helper also measures the activation spread after each ReLU, the quickest health check there is.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

def make_mlp(depth=20, width=256, d_in=784, d_out=10):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)

def init_he(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # std = sqrt(2 / fan_in)
            nn.init.zeros_(m.bias)

@torch.no_grad()
def activation_stds(model, x):
    stds = []
    for layer in model:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            stds.append(round(x.std().item(), 3))
    return stds

x = torch.randn(512, 784)
default = make_mlp()                      # PyTorch's built-in init
he = make_mlp(); init_he(he)
print("default:", activation_stds(default, x)[::5])
print("He:     ", activation_stds(he, x)[::5])
# default: [0.336, 0.022, 0.021, 0.023]
# He:      [0.829, 0.947, 1.301, 1.338]

w = nn.Linear(784, 256).weight
print(f"default std {w.std():.4f}, He target {(2/784)**0.5:.4f}")
# default std 0.0206, He target 0.0505
```

PyTorch's default weight standard deviation is $1/\sqrt{3 \times 784} = 0.0206$, well under He's 0.0505. With the default, activations settle around 0.02 rather than reaching zero because the default also draws **biases** at random (uniform in $\pm 1/\sqrt{n_{in}}$), which keep injecting a small signal; by then most of the input's information is gone. He keeps the spread near 1.

## Does it change training?

Statistics are a proxy; the real test is learning. A plain 30-layer ReLU MLP (width 64, no normalisation, no skip connections) is trained on scikit-learn's 8×8 digits with identical optimiser, mini-batches and seed; only the initial weights differ.

~~~dg-fig 01.05-train.svg
Training loss for four initialisations of the same 30-layer ReLU network. Only He gets off the ln 10 line: after 1000 steps it is at loss 0.01 and 99.7% training accuracy, while Xavier, the PyTorch default and N(0, 0.01²) are still guessing.
~~~

The optimiser is SGD (learning rate 0.002, momentum 0.9) for 1000 mini-batches of 128. The dashed line at $\ln 10 = 2.30$ is the loss of equal probability on all 10 digits. Three of the four never leave it:

- **N(0, 0.01²)** and **PyTorch's default** end at 10% training accuracy, pure guessing; their gradients near the input were 0 and $10^{-12}$.
- **Xavier** finishes at loss 2.30 and 17% accuracy.
- **He** is below 1.0 within about 200 steps and finishes at loss 0.01, 99.7% training accuracy.

He et al. reported the same effect for a 30-layer plain network: Xavier stalled, their rule trained. One caveat: normalisation layers (02.01, 02.02) and residual connections (03.04) make networks far less sensitive to the starting scale, which is why PyTorch can get away with a conservative default. They do not remove the need for symmetry breaking.

## Reading the diagnostics

Before training anything deep, run one batch and look at three numbers per layer:

- **Activation standard deviation after each nonlinearity.** It should stay within a small factor of 1. A steady fall or rise by a constant ratio per layer means the gain is off by that ratio.
- **Fraction of dead or saturated units.** For ReLU, the share of units that output 0 for every example in the batch; for tanh or sigmoid, the share beyond ±0.95 or so. Either above roughly half is a warning.
- **Gradient norm per layer after one backward pass.** Roughly flat is healthy; orders of magnitude between first and last layer means early layers will not learn (or will blow up).

A fourth check: the first loss should be about $\ln C$ for $C$ classes (2.30 for 10). A first loss of 40 means the output layer starts too large.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| All weights equal (often zero) | Hidden units stay identical; accuracy capped at what one unit can do | Random, zero-mean weights; biases may be zero
| Weights from `torch.randn` (σ = 1) | Activations explode (ReLU) or saturate at ±1 (tanh); NaN loss | Scale by $1/\sqrt{n_{in}}$ or use `nn.init`
| Fixed small σ like 0.01 in a deep net | Activations and gradients tiny or exactly 0; loss stuck at $\ln C$ | He for ReLU, Xavier/LeCun for tanh
| Xavier or default on a deep plain ReLU net | Second moment shrinks 2× (Xavier) or 6× (default) per layer; no learning | `kaiming_normal_(..., nonlinearity="relu")`
| Fan-in computed wrongly for conv | Off by the factor $k^2$ | $n_{in} = C_{in} k^2$
| Large output layer | First loss far above $\ln C$, confident wrong predictions | Shrink or zero-init the last layer

## Going deeper

**Why "about 1" is not quite enough.** The variance argument tracks the average size of signals, but a product of many random matrices can stretch some directions and squash others even when the average gain is exactly 1. Saxe et al. (2014) proposed **orthogonal initialisation**: each square weight matrix is a random rotation (times the gain), so a linear network preserves every direction exactly. Pennington et al. (2017) call the goal **dynamical isometry**: all singular values of the input-to-output Jacobian near 1, not just their average. It is achievable with tanh plus orthogonal weights, but not with ReLU, which zeroes half the directions.

**Order, chaos and the edge between.** Mean-field analyses (Poole et al. 2016, Schoenholz et al. 2017) treat a wide random tanh network as a dynamical system in depth. Small weights give an **ordered** phase (all inputs flow to the same output, gradients vanish); large weights give a **chaotic** phase (nearby inputs are pushed apart, gradients explode). Good initialisation sits on the boundary. This is why tanh with $\sigma_w = 1$ (the right panel of the tanh figure) is not just slow: in our 40-layer, width-256 test its first-layer gradient norm was about $10^{16}$ while its last layer's was 0.01. He on tanh is mildly chaotic (first-layer gradient 3.5 times the last), Xavier close to critical.

**Residual networks.** A residual block outputs $x + F(x)$, so variances add and the signal grows with the number of blocks even if each branch is well initialised. Fixes: zero-init the scale $\gamma$ of each branch's last normalisation layer so every block starts as the identity (Goyal et al. 2017), or scale each branch's output projection by $1/\sqrt{N}$ for $N$ residual layers, as GPT-2 does (nanoGPT: std $0.02/\sqrt{2 \cdot n_{layer}}$). Fixup (Zhang et al. 2019) trains deep ResNets without normalisation using such rescaling.

**Width and learning rates.** These rules fix the scale at step zero only. How learning rate and initial scale should change with width is answered by muP (the Maximal Update Parametrisation); the practice problem on muP width invariants is a starting point.

## Key terms

| Term | Meaning
| --- | --- |
| Weight initialisation | The rule for drawing the starting weights and biases before training
| Symmetry breaking | Using random weights so hidden units start different and can learn different features
| Fan-in / fan-out | Number of inputs into one unit / number of outputs one input feeds; for conv, fan-in is $C_{in} k^2$
| Per-layer gain | Factor by which one layer multiplies the signal's second moment; must be close to 1
| LeCun init | $\mathrm{Var}(w) = 1/n_{in}$; for linear or SELU units
| Xavier (Glorot) init | $\mathrm{Var}(w) = 2/(n_{in}+n_{out})$; compromise between forward and backward, for tanh/sigmoid
| He (Kaiming) init | $\mathrm{Var}(w) = 2/n_{in}$; doubles LeCun to make up for ReLU zeroing half its inputs
| Saturation | Units stuck in the flat part of tanh or sigmoid, with derivative near 0

<div class="take">
<div class="lab">Key takeaway</div>

draw weights at random to break symmetry, and choose their variance so each layer's gain is about 1: $2/n_{in}$ (He) for ReLU, $2/(n_{in}+n_{out})$ (Xavier) or $1/n_{in}$ for tanh. Then check one batch: activation spread, dead or saturated units, per-layer gradient norms and the first loss tell you whether the start is sane before you spend a single training run.

</div>
