---
id: "01.01"
course: "deep-learning"
module: "01"
slug: "neurons-layers-and-the-mlp"
title: "Neurons, layers and the MLP"
keyIdeas: ["A neuron computes an affine function of its inputs followed by a nonlinearity, $h = \\phi(w^\\top x + b)$. Geometrically a ReLU neuron is a line (hyperplane): zero on one side, a rising ramp on the other.", "A layer is many neurons sharing the same input, i.e. one matrix multiply plus an elementwise $\\phi$: $H = \\phi(XW^\\top + b)$ for a batch $X$ of shape $B \\times n_{in}$. An MLP stacks such layers; the last one outputs raw logits.", "Without the nonlinearity any stack of layers collapses to a single linear map $Mx + c$, so depth buys nothing: on two moons the same 609-parameter network reaches 85% without ReLU and 100% with it.", "The universal approximation theorem says one wide hidden layer can approximate any continuous function on a bounded domain, but it says nothing about how wide, how to find the weights, or how well the result generalises.", "Depth helps because composition reuses features: a sawtooth with $2^k$ pieces needs only $2k$ units over $k$ layers but about $2^k$ units in one hidden layer.", "Count parameters and shapes before anything else: a dense layer from $n_{in}$ to $n_{out}$ has $n_{in} n_{out} + n_{out}$ parameters and costs about $n_{in} n_{out}$ multiply-adds per example; training costs roughly 3 times the forward pass."]
question: "A colleague removes every activation from a 10-layer, 512-wide MLP \"to make it train faster\" and finds training accuracy drops to that of logistic regression. Explain exactly why, say whether adding more layers or more width would recover the lost accuracy, and estimate how many parameters the activation-free network wastes if its input has 784 features and its output 10 classes (10 weight layers: 784 to 512, eight 512 to 512, 512 to 10)."
answer: "Without activations each layer is affine, and a composition of affine maps is affine: $W_{10}(\\cdots(W_1 x + b_1)\\cdots) + b_{10}$ simplifies to $Mx + c$ with $M = W_{10}\\cdots W_1$ (shape $10 \\times 784$) and a single bias vector $c$. The network therefore represents exactly the same function class as one linear layer followed by softmax, which is multinomial logistic regression, so its decision boundaries between classes are hyperplanes no matter how many parameters it has. Adding layers cannot help, because every extra linear layer is absorbed into $M$; adding width cannot help either, since a product of matrices can never have more than rank 10 here and the reachable set of maps is unchanged. At best extra layers change the optimisation dynamics (deep linear networks have non-convex landscapes and can train more slowly), never the expressive power. Parameter count: $784 \\cdot 512 + 512 = 401{,}920$; eight hidden-to-hidden layers of $512 \\cdot 512 + 512 = 262{,}656$ each give $2{,}101{,}248$; the output layer has $512 \\cdot 10 + 10 = 5{,}130$. Total $2{,}508{,}298$ parameters, while the equivalent logistic regression needs only $784 \\cdot 10 + 10 = 7{,}850$, so about 2.5 million parameters (over 99.6%) do nothing except slow training down. Accuracy comes back only by restoring a nonlinearity such as ReLU between the affine maps, which lets each hidden layer bend and fold the representation so later layers can separate classes that are not linearly separable in the input; to confirm the fix works, overfit a small batch and compare held-out accuracy with the logistic-regression baseline."
sources: [{"title": "Deep Learning (Goodfellow, Bengio, Courville), Chapter 6", "url": "https://www.deeplearningbook.org/contents/mlp.html", "note": "Sections 6.1 and 6.4 on XOR, universal approximation and the argument for depth."}, {"title": "Dive into Deep Learning: Multilayer Perceptrons", "url": "https://d2l.ai/chapter_multilayer-perceptrons/mlp.html", "note": "Short derivation of why hidden layers need nonlinearities, with runnable code."}, {"title": "Telgarsky: Benefits of depth in neural networks", "url": "https://arxiv.org/abs/1602.04485", "note": "The sawtooth construction showing exponential separation between deep and shallow nets."}, {"title": "CS231n: Neural Networks Part 1", "url": "https://cs231n.github.io/neural-networks-1/", "note": "Neuron model, layer sizing and the representational-power discussion."}]
practice: ["multi-layer-perceptron-forward-pass", "evaluate-multi-layer-perceptron", "simple-neural-network", "evaluate-dense-layer", "perceptron"]
tags: ["MLP", "universal approximation", "depth"]
tools: ["NumPy"]
---

## What you will learn

- What an artificial neuron computes: a weighted sum, a bias, and a nonlinearity, and how to picture it as a line that splits the input space.
- How many neurons side by side become a **layer** (one matrix multiply), and how layers stacked in sequence become a **multilayer perceptron (MLP)**.
- Why the nonlinearity is not optional: without it, any number of layers collapses to a single linear model.
- What the universal approximation theorem promises, what it does not, and why depth still helps.
- How to track shapes, count parameters and estimate compute for any dense network, by hand and in NumPy / PyTorch.

*Before you start:* you should be comfortable with a linear model $\hat y = w^\top x + b$ and with logistic regression (Machine Learning course, lessons 02.01 and 02.03), and with multiplying a matrix by a vector (ML Maths, module 08). Gradients and optimisers are used here as black boxes; lesson 01.04 opens the box.

## Where this module starts

This module builds a neural network from its smallest parts: the neuron, layer and MLP (this lesson), activation functions (01.02), losses (01.03), the backward pass (01.04) and initial weights (01.05). Module 02 covers what keeps deep networks trainable, and Module 03 applies it all to convolutional networks. Optimiser update rules (SGD, momentum, Adam) live in ML Maths; here they are tools you call.

## The idea in one picture

Take the smallest problem a linear model cannot solve: **XOR**. Two binary inputs, and the label is 1 when exactly one of them is 1.

| $x_1$ | $x_2$ | label
| --- | --- | --- |
| 0 | 0 | 0
| 0 | 1 | 1
| 1 | 0 | 1
| 1 | 1 | 0

The label-1 points sit on one diagonal and the label-0 points on the other (left panel below). No straight line separates them, so logistic regression, or any single linear model, must get at least one point wrong.

Now add one **hidden layer** of two units. Each unit computes a weighted sum of the inputs, adds a bias, and then applies **ReLU**, the function $\text{ReLU}(z) = \max(0, z)$ that keeps positive numbers and replaces negative ones with 0:

$$h_1 = \text{ReLU}(x_1 + x_2), \qquad h_2 = \text{ReLU}(x_1 + x_2 - 1).$$

These two numbers re-describe each point, and in that description (right panel) the picture changes completely.

~~~dg-fig 01.01-xor.svg
XOR before and after one hidden ReLU layer. Left: no line separates squares from circles. Right: the hidden layer moves (0,1) and (1,0) onto the same spot and pushes (1,1) away, so one dashed line now does the job.
~~~

Point (1,1) moves to $(2, 1)$, while (0,1) and (1,0) both land on $(1, 0)$. A single output unit $\hat y = h_1 - 2h_2$ now reads off the answer. That is the whole idea of a neural network in miniature: hidden layers **re-describe** the input so that a simple linear rule at the end becomes enough.

Words you will use all the time:

- A **neuron** (or **unit**) computes a weighted sum of its inputs plus a bias, then applies a nonlinearity.
- The **activation function** is that nonlinearity ($\phi$ in formulas). ReLU is the default today; lesson 01.02 compares the alternatives.
- A **hidden layer** is any layer whose outputs are neither the inputs nor the final prediction. Its outputs are called **hidden features** or **activations**.

## Step 1: one neuron

A neuron takes an input vector $x$ with $n$ numbers and computes

$$z = w^\top x + b = w_1 x_1 + \dots + w_n x_n + b, \qquad h = \phi(z).$$

Here $w$ is the **weight vector** (one weight per input), $b$ is the **bias** (one number), $z$ is the **pre-activation** and $h$ is the neuron's output. The first part is exactly a linear model. The only new thing is $\phi$.

Two classic special cases:

- With $\phi$ = a step function (output 1 if $z > 0$, else 0) this is Rosenblatt's **perceptron** (1958).
- With $\phi$ = the sigmoid $1/(1+e^{-z})$ it is logistic regression.

**Picture it.** For two inputs, $z = 0$ is a straight line in the plane, and $w$ points across it towards where $z$ grows. A ReLU neuron outputs exactly 0 on one side and a steadily rising ramp on the other.

~~~dg-fig 01.01-neuron.svg
One ReLU neuron with w = (2, 1) and b = −2. In the shaded half-plane the output is exactly zero; across the red boundary it grows linearly in the direction of w, one unit per contour line.
~~~

Number check with this neuron: at $x = (1, 1)$, $z = 2 \cdot 1 + 1 \cdot 1 - 2 = 1$, so $h = 1$. At $x = (0, 1)$, $z = 0 + 1 - 2 = -1$, so $h = 0$: the neuron is "off". A neuron is a **feature detector**: it answers "how far past my line is this input?" and is silent on the other side.

## Step 2: a layer is one matrix multiply

A **layer** is $m$ neurons that all read the same input. Each has its own weight vector and bias. Stack the $m$ weight vectors as the rows of a matrix $W$ (shape $m \times n$) and the biases into a vector $b$ (length $m$). Then all $m$ neurons at once are

$$h = \phi(Wx + b), \qquad W: m \times n, \; b: m, \; h: m.$$

Row $j$ of $W$ is neuron $j$'s weights, and $\phi$ acts **elementwise**, on each of the $m$ numbers separately. Since every input connects to every output, this is a **dense** or **fully connected** layer (`nn.Linear` in PyTorch, `Dense` in Keras).

In the XOR network $W = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$ and $b = (0, -1)$; for $x = (1, 1)$, $Wx + b = (2, 1)$, the same as before in one operation.

**Batches.** In practice $B$ examples go through together, as rows of a matrix $X$ (shape $B \times n$):

$$H = \phi(X W^\top + b), \qquad X: B \times n, \; W^\top: n \times m, \; H: B \times m,$$

where $b$ is added to every row (broadcasting). One matrix multiply handles the whole batch, which is exactly what GPUs are built for. Getting these shapes right is most of the work in debugging a network.

## Step 3: stack layers into an MLP

A **multilayer perceptron** is several dense layers in sequence, each feeding the next:

$$h^{(1)} = \phi(W^{(1)}x + b^{(1)}), \quad h^{(2)} = \phi(W^{(2)}h^{(1)} + b^{(2)}), \quad \dots, \quad \text{output} = W^{(L)}h^{(L-1)} + b^{(L)}.$$

The superscript $(l)$ numbers the layers. The last layer has **no** activation: its raw outputs, the **logits**, go to the loss, which applies softmax or sigmoid itself (lesson 01.03). The **depth** of the network is the number of layers with weights; the **width** of a layer is its number of units. Computing the output from the input, layer by layer, is the **forward pass**.

~~~dg-flow A two-layer MLP for 10-class classification of 28×28 images: affine, nonlinearity, affine, then the loss turns logits into a number.
x: B×784 -(W1, b1)-> z1: B×256 -(ReLU)-> h1: B×256 -(W2, b2)-> *logits: B×10 -(softmax + CE)-> loss
~~~

<div class="ex">
<div class="lab">Worked example</div>

**forward pass of the XOR network by hand**

Network: hidden layer $W^{(1)} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, $b^{(1)} = (0, -1)$, ReLU; output layer $w^{(2)} = (1, -2)$, $b^{(2)} = 0$, no activation. Predict label 1 when the output is above 0.5.

Input (0, 0): $z^{(1)} = (0 + 0 + 0,\; 0 + 0 - 1) = (0, -1)$. ReLU gives $h = (0, 0)$. Output $= 1 \cdot 0 - 2 \cdot 0 = 0$. Predict 0. Correct.

Input (0, 1): $z^{(1)} = (1, 0)$, $h = (1, 0)$, output $= 1 - 0 = 1$. Predict 1. Correct.

Input (1, 0): by symmetry $h = (1, 0)$ and output $= 1$. Predict 1. Correct.

Input (1, 1): $z^{(1)} = (2, 1)$, $h = (2, 1)$, output $= 2 - 2 \cdot 1 = 0$. Predict 0. Correct.

All four right, with 9 parameters (4 + 2 hidden, 2 + 1 output). Unit $h_2$ switches on only when both inputs are 1, and its weight of $-2$ cancels $h_1$'s over-count.

</div>

## Step 4: why the nonlinearity is mandatory

Suppose you delete $\phi$ to save time. Two linear layers then compose to

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\,x + (W_2 b_1 + b_2) = M x + c.$$

That is one linear layer with matrix $M = W_2 W_1$ and bias $c$. Repeat the argument and a 100-layer linear network has exactly the expressive power of logistic regression, only with a different, usually worse, optimisation landscape.

The figure below trains one network twice on two moons (300 points, two interleaved half-circles): three hidden layers of 16 units plus an output unit, 609 parameters, 1500 Adam steps. The only difference is whether ReLU sits between the layers.

~~~dg-fig 01.01-moons.svg
Same 609-parameter network, same data and training. Without activations (left) the boundary is a straight line and training accuracy stops at 85%, the level of logistic regression. With ReLU (right) it bends around the moons and reaches 100%. Shading marks the region predicted as class 1.
~~~

With ReLU, each hidden unit adds a fold along its own line, as in the neuron figure, so the output is **piecewise linear**: flat pieces joined at creases, hence the straight segments in the right-hand boundary. More units and layers mean more pieces.

## The MLP in code

A forward pass in NumPy takes a few lines. Note the convention: `W` has shape `(n_out, n_in)`, the same as PyTorch's `nn.Linear.weight`.

```python
import numpy as np

def mlp_forward(X, params):
    # params: list of (W, b) with W shaped (n_out, n_in)
    h = X
    for i, (W, b) in enumerate(params):
        z = h @ W.T + b
        h = np.maximum(z, 0) if i < len(params) - 1 else z   # no ReLU on logits
    return h

rng = np.random.default_rng(0)
sizes = [784, 256, 10]
params = [(rng.normal(0, np.sqrt(2 / a), (b_, a)), np.zeros(b_)) for a, b_ in zip(sizes, sizes[1:])]
print(mlp_forward(rng.normal(size=(32, 784)), params).shape)   # (32, 10)
print(sum(W.size + b.size for W, b in params))                 # 203530
```

(The `np.sqrt(2 / a)` scale is He initialisation, explained in lesson 01.05.) In PyTorch you describe the same structure with `nn.Sequential`, and autograd handles the gradients. This is the network behind the moons figure:

```python
import torch, torch.nn as nn
from sklearn.datasets import make_moons

X, y = make_moons(300, noise=0.2, random_state=0)
X = torch.tensor(X, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32)

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 16), nn.ReLU(),
    nn.Linear(16, 16), nn.ReLU(),
    nn.Linear(16, 1),                       # logits: no activation here
)
print(sum(p.numel() for p in model.parameters()))   # 609
print(model[0].weight.shape)                        # torch.Size([16, 2])  (out, in)

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(1500):
    logits = model(X).squeeze(1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
acc = ((model(X).squeeze(1) > 0).float() == y).float().mean()
print(round(loss.item(), 3), round(acc.item(), 3))  # 0.007 1.0
```

Training accuracy of 1.0 on noisy data shows flexibility, not quality: some points sit on the wrong side of the true boundary, and fitting them is memorisation. Judge on held-out data (Module 02).

## Counting parameters and compute

Before training anything, write down the shapes and count. A dense layer from $n_{in}$ to $n_{out}$ units has

$$n_{in} \cdot n_{out} \text{ weights} + n_{out} \text{ biases},$$

and its forward pass costs about $n_{in} \cdot n_{out}$ **multiply-adds** (MACs) per example. The backward pass costs about twice that (one matrix multiply for the input gradient, one for the weight gradient), so a training step is roughly 3 times the forward compute.

Check against the moons network: $2 \cdot 16 + 16 = 48$, then $16 \cdot 16 + 16 = 272$ twice, then $16 \cdot 1 + 1 = 17$. Total $48 + 272 + 272 + 17 = 609$, matching PyTorch.

<div class="ex">
<div class="lab">Worked example</div>

**sizing an MLP for 28×28 digit images**

Input: 28 × 28 = 784 pixels, flattened into one vector. Hidden layer 256 units, output 10 classes.

Layer 1: 784 × 256 = 200,704 weights + 256 biases = 200,960 parameters.

Layer 2: 256 × 10 = 2,560 weights + 10 biases = 2,570 parameters.

Total: 200,960 + 2,570 = 203,530 parameters (the NumPy code above prints the same). At 4 bytes each in fp32 that is 814,120 bytes, about 0.8 MB.

Forward cost: about 200,704 + 2,560 = 203,264 MACs per image; a batch of 128 is about 26 million MACs, trivial for any modern processor.

Notice that 200,960 / 203,530 ≈ 98.7% of the parameters sit in the first layer: wide inputs dominate MLP cost, one reason images moved to convolutions (Module 03).

</div>

## Universal approximation, and what it does not say

How flexible is an MLP? The **universal approximation theorem** (Cybenko 1989 for sigmoids; Hornik and others, and Leshno et al. 1993, for any non-polynomial activation including ReLU) says: a network with **one** hidden layer and enough units can approximate any continuous function on a bounded domain to any accuracy you like.

For ReLU in 1-D you can see why directly: one unit per knot adds a crease there, and setting output weights to the changes in slope traces the straight-line interpolation of the target. More units, more knots, smaller error.

~~~dg-fig 01.01-approx.svg
A one-hidden-layer ReLU network built by hand to approximate f(x) = sin(2x) + 0.3x on [−3, 3]. With 3, 8 and 30 units the largest error falls from 1.15 to 0.23 to 0.02: width buys accuracy.
~~~

Three caveats make the theorem less useful than it sounds:

- **Width can be enormous.** The theorem only says suitable weights exist; the units needed can grow exponentially with input dimension.
- **It says nothing about learning.** Gradient descent may never find the weights that exist.
- **It says nothing about generalisation.** Fitting the training points is not the same as predicting new ones.

## Why depth helps

If one hidden layer suffices in principle, why use many? Because depth is a far cheaper source of flexibility.

- **Compositional reuse.** Each layer builds on the previous layer's features: in images, edges, then textures, then parts, then objects. A shallow network must build every high-level feature from raw pixels independently.
- **Exponential efficiency for some functions.** Take the "tent" $t(x) = 2\,\text{ReLU}(x) - 4\,\text{ReLU}(x - 0.5)$, which rises from 0 to 1 and back on $[0, 1]$ using two units. Feed its output into another copy of itself, and again.

~~~dg-fig 01.01-sawtooth.svg
Composing the same two-unit tent 1, 2, 3 and 4 times gives 2, 4, 8 and 16 linear pieces. Depth doubles the pieces per layer; one hidden layer would need roughly one unit per crease, here 15.
~~~

With $k$ layers of 2 units each the result has $2^k$ pieces. A one-hidden-layer ReLU network on a 1-D input gets at most one new crease per unit, so it needs at least $2^k - 1$ units to draw the same sawtooth. Telgarsky (2016) turned this into a theorem: there are functions computed by networks with about $k^3$ layers of constant width that networks with only about $k$ layers cannot approximate unless they have on the order of $2^k$ units.

- **Parameter efficiency.** At a fixed budget, deeper-and-narrower usually beats shallow-and-wide on real data.

Depth has a price: gradients must travel back through every layer and can shrink or blow up on the way. Initialisation (01.05), most of Module 02, and the residual connection in 03.04 exist to pay that price.

## Reading / checking the result

Before trusting a network you just wrote, check in order:

- **Shapes.** Push one small batch through and print every intermediate shape. The batch dimension must survive, and the final width must equal the number of outputs.
- **Parameter count.** Compare your hand count with `sum(p.numel() for p in model.parameters())`. A mismatch means a layer is wired differently from what you think.
- **Untrained loss.** A freshly initialised 10-class classifier should give a loss near $\ln 10 \approx 2.30$ (roughly uniform guesses). Much higher means badly scaled weights or inputs.
- **Overfit a tiny batch.** A working network drives the loss on 10-20 examples to nearly zero within a few hundred steps. If it cannot, there is a bug (missing activation, wrong loss input, frozen parameters).

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| No activation between layers | Accuracy stuck at logistic-regression level; straight-line boundaries | Put ReLU (or another nonlinearity) after every hidden layer
| Activation on the output layer | ReLU on logits clips negatives; loss plateaus | Last layer returns raw logits; the loss applies softmax / sigmoid
| Applying softmax and then `CrossEntropyLoss` | Training slow or stuck, loss never near 0 | PyTorch's cross-entropy expects logits; drop the extra softmax
| Weight shape confusion | Shape-mismatch errors, or silent nonsense when $n_{in} = n_{out}$ | PyTorch stores $W$ as (out, in) and computes $XW^\top$; Keras stores (in, out)
| Unscaled inputs | Some units always off, others saturated; slow training | Standardise features to mean 0, variance 1 before the first layer
| Network too narrow | Training loss high, curves flat | Widen or deepen; check by overfitting a tiny batch

## Going deeper

**Linear regions.** A ReLU network computes a continuous piecewise-linear function. Each hidden unit's line (a hyperplane in $n$ dimensions) is a potential crease. One hidden layer of $m$ units on $n$ inputs is an arrangement of $m$ hyperplanes, which has at most $\sum_{i=0}^{n}\binom{m}{i}$ regions: polynomial in $m$. Stacking layers lets later units fold the regions created by earlier ones, so the maximum count can grow exponentially with depth, as the sawtooth shows. Trained networks typically use far fewer regions than the maximum: this is about capacity, not about what training finds.

**Sketch of the 1-D approximation argument.** Let $f$ be continuous on $[a, c]$ and pick knots $a = k_0 < k_1 < \dots < k_N = c$. With slopes $s_j = (f(k_{j+1}) - f(k_j))/(k_{j+1} - k_j)$, the function

$$g(x) = f(k_0) + s_0\,\text{ReLU}(x - k_0) + \sum_{j=1}^{N-1} (s_j - s_{j-1})\,\text{ReLU}(x - k_j)$$

equals the straight-line interpolation of $f$ through the knots, using $N$ hidden units. Because $f$ is uniformly continuous on a closed interval, shrinking the spacing makes $\max|g - f|$ as small as you like; this is the construction behind the approximation figure. In $n$ dimensions the same idea needs a grid with exponentially many cells: the "width can be enormous" caveat made concrete.

**Symmetry.** Permuting the hidden units of a layer (rows of $W^{(l)}$ and $b^{(l)}$, columns of $W^{(l+1)}$) leaves the function unchanged, so a width-$m$ layer has $m!$ equivalent weight settings. It is also why weights must not start equal: identical units get identical gradients and stay identical (lesson 01.05).

AI Research's "Neural Network from Scratch" lessons build these pieces in code step by step; this lesson is the conceptual frame for the rest of this course.

## Key terms

| Term | Meaning
| --- | --- |
| Neuron / unit | Weighted sum of inputs plus a bias, passed through an activation function
| Activation function | The elementwise nonlinearity $\phi$, such as ReLU $= \max(0, z)$
| Dense (fully connected) layer | $m$ neurons reading the same input: $h = \phi(Wx + b)$ with $W$ of shape $m \times n$
| Hidden layer | A layer between input and output whose activations are learned features
| MLP | Dense layers stacked with activations between them; the last layer outputs logits
| Depth / width | Number of weight layers / number of units in a layer
| Universal approximation | One wide hidden layer can approximate any continuous function on a bounded domain
| MAC | One multiply-add; a dense layer costs $n_{in} n_{out}$ per example

<div class="take">
<div class="lab">Key takeaway</div>

an MLP alternates matrix multiplies with elementwise nonlinearities; the nonlinearity is what lets hidden layers re-describe the input so a simple rule works at the end, and depth makes that flexibility cheap, at the price of harder optimisation.

</div>
