---
id: "01.01"
course: "deep-learning"
module: "01"
slug: "neurons-layers-and-the-mlp"
title: "Neurons, layers and the MLP"
keyIdeas: ["A neuron computes an affine function of its inputs followed by a nonlinearity, $h = \\phi(w^\\top x + b)$; a layer is many neurons sharing the same input, i.e. one matrix multiply plus an elementwise $\\phi$.", "Without the nonlinearity, any stack of layers collapses to a single linear map, so depth buys nothing.", "The universal approximation theorem says one wide hidden layer can approximate any continuous function on a compact set, but it says nothing about how wide, how to find the weights, or how well it generalises.", "Depth helps because composition reuses features: some functions need exponentially many units with one hidden layer but only a modest number with several layers.", "Count parameters and shapes before anything else: a dense layer from $n_{in}$ to $n_{out}$ has $n_{in} n_{out} + n_{out}$ parameters and costs about $n_{in} n_{out}$ multiply-adds per example."]
question: "A colleague removes every activation from a 10-layer, 512-wide MLP \"to make it train faster\" and finds training accuracy drops to that of logistic regression. Explain exactly why, and whether adding more layers would recover the lost accuracy."
answer: "Without activations each layer is affine, and a composition of affine maps is affine: $W_{10}\\cdots W_1 x + c$ collapses to a single matrix $M$ and bias. The network therefore represents exactly the same function class as a single linear layer followed by softmax, which is multinomial logistic regression. Its decision boundaries are hyperplanes no matter how many parameters it has. Adding layers cannot help, because every extra linear layer is absorbed into $M$; at best it changes the optimisation dynamics (deep linear nets have non-convex landscapes and can train more slowly), never the expressive power. Accuracy comes back only by restoring a nonlinearity such as ReLU between the affine maps, which lets each layer bend the representation so later layers can separate classes that are not linearly separable in the input."
sources: [{"title": "Deep Learning (Goodfellow, Bengio, Courville), Chapter 6", "url": "https://www.deeplearningbook.org/contents/mlp.html", "note": "Sections 6.1 and 6.4 on XOR, universal approximation and the argument for depth."}, {"title": "Dive into Deep Learning: Multilayer Perceptrons", "url": "https://d2l.ai/chapter_multilayer-perceptrons/mlp.html", "note": "Short derivation of why hidden layers need nonlinearities, with runnable code."}, {"title": "Telgarsky: Benefits of depth in neural networks", "url": "https://arxiv.org/abs/1602.04485", "note": "The sawtooth construction showing exponential separation between deep and shallow nets."}, {"title": "CS231n: Neural Networks Part 1", "url": "https://cs231n.github.io/neural-networks-1/", "note": "Neuron model, layer sizing and the representational-power discussion."}]
practice: ["multi-layer-perceptron-forward-pass", "evaluate-multi-layer-perceptron", "simple-neural-network", "evaluate-dense-layer", "perceptron"]
tags: ["MLP", "universal approximation", "depth"]
tools: ["NumPy"]
---

## Where this course starts

This module builds a neural network from its smallest parts: the neuron, the activation, the loss, the backward pass that computes gradients, and the initial weights that let training start at all. Module 02 then covers what keeps deep networks trainable (normalisation, regularisation, schedules, diagnostics), and Module 03 applies all of it to convolutional networks. The maths of gradients and the optimiser update rules themselves (SGD, momentum, Adam) live in ML Maths; here they are tools.

## One neuron, one layer

A neuron takes a vector $x \in \mathbb{R}^{n}$, forms a weighted sum plus a bias, and passes it through a nonlinearity $\phi$:

$$h = \phi(w^\top x + b)$$

With $\phi$ = step function this is Rosenblatt's perceptron; with $\phi$ = sigmoid it is logistic regression. A **layer** is $m$ neurons reading the same input. Stack their weight vectors as rows of $W \in \mathbb{R}^{m \times n}$ and the whole layer is one matrix multiply:

$$h = \phi(Wx + b), \quad W: m \times n, \; b: m$$

In code you process a batch of $B$ examples at once, so inputs are a $B \times n$ matrix and the layer computes `X @ W.T + b`, giving $B \times m$. Getting these shapes right is most of the work in debugging a network.

~~~dg-flow A two-layer MLP for 10-class classification: affine, nonlinearity, affine, then the loss turns logits into a number.
x: B×784 -(W1, b1)-> z1: B×256 -(ReLU)-> h1: B×256 -(W2, b2)-> *logits: B×10 -(softmax + CE)-> loss
~~~

## Why the nonlinearity is mandatory

Two linear layers compose to one: $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$. A 100-layer linear network has exactly the expressive power of logistic regression; it only has a different (and worse) optimisation landscape. The elementwise $\phi$ breaks this. With ReLU, each hidden unit splits input space with a hyperplane, and the network becomes a piecewise-linear function whose number of linear pieces grows with width and, much faster, with depth.

## Universal approximation, and what it does not say

The universal approximation theorem (Cybenko 1989 for sigmoids, Hornik and others for general non-polynomial activations) states that a network with **one** hidden layer and enough units can approximate any continuous function on a bounded domain to any accuracy. Three caveats make it less useful than it sounds:

- **Width can be enormous.** The theorem is existential; the number of units needed can grow exponentially with input dimension.
- **It says nothing about learning.** Gradient descent may never find the weights that exist.
- **It says nothing about generalisation.** Fitting the training points is not the same as predicting new ones.

## Why depth helps

Deep networks win in practice for reasons the theorem doesn't cover:

- **Compositional reuse.** Layer 2 builds on layer 1's features, layer 3 on layer 2's. In images: edges, then textures, then parts, then objects. A shallow net must build every high-level feature from raw pixels independently.
- **Exponential efficiency for some functions.** Telgarsky showed functions computable by a network with $O(k^3)$ layers and constant width that need on the order of $2^k$ units if the network has only $O(k)$ layers. The classic intuition: a ReLU "tent" composed with itself $k$ times makes a sawtooth with $2^k$ teeth using $O(k)$ units, while one hidden layer needs a unit per tooth.
- **Parameter efficiency.** At a fixed parameter budget, deeper-and-narrower usually beats shallow-and-wide on real data.

Depth has a cost: gradients must travel through many layers and can vanish or explode. Most of Module 02 and the residual connection in 03.04 exist to pay that cost.

## Counting parameters and compute

A dense layer $n_{in} \to n_{out}$ has $n_{in} n_{out}$ weights plus $n_{out}$ biases, and costs $n_{in} n_{out}$ multiply-adds (MACs) per example in the forward pass. The backward pass costs about twice that (one matmul for the input gradient, one for the weight gradient), so training is roughly 3× forward compute.

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
```

<div class="ex">
<div class="lab">Worked example</div>

**sizing an MNIST MLP**

Layer 1: 784 × 256 weights + 256 biases = 200,960 parameters. Layer 2: 256 × 10 + 10 = 2,570. Total 203,530 parameters, about 0.8 MB in fp32.

Forward cost per image is about 203k MACs; a batch of 128 is about 26 million MACs, trivial for any GPU. Notice 99% of parameters sit in the first layer: wide inputs dominate MLP cost, which is one reason images moved to convolutions (Module 03).

</div>

| Question | Answer |
| --- | --- |
| What does one hidden layer guarantee? | Approximation exists, given enough units |
| What does it not guarantee? | Reasonable width, learnability, generalisation |
| Why go deeper? | Feature reuse, exponential savings on compositional functions |
| What does depth cost? | Harder gradient flow; needs init, normalisation, residuals |

<div class="take">
<div class="lab">Key takeaway</div>

an MLP is alternating affine maps and elementwise nonlinearities; the nonlinearity is what gives it power, and depth is what makes that power cheap, at the price of harder optimisation.

</div>

AI Research's "Neural Network from Scratch" lessons build these pieces in code step by step; this lesson is the conceptual frame for the rest of the course.
