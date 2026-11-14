---
id: "01.04"
course: "deep-learning"
module: "01"
slug: "backpropagation-through-a-computational-graph"
title: "Backpropagation through a computational graph"
keyIdeas: ["Any computation built from simple operations (add, multiply, tanh, matmul) is a **computational graph**, a DAG; each node only needs its own local derivative rule to take part in backprop.", "Backprop is two passes: the forward pass computes and stores every intermediate value; the backward pass visits nodes in reverse topological order and multiplies each upstream gradient by the local derivative (the chain rule).", "When a value feeds several operations, its gradient is the **sum** over all paths, so backward rules accumulate with `+=`; for the same reason PyTorch adds into `.grad` and you must call `zero_grad()` every step.", "Test a hand-written backward with central finite differences in float64: relative errors near 1e-7 or below mean correct, near 1e-2 mean a bug; float32 checks give false alarms.", "Reverse mode gives the gradient for any number of parameters at about 2-3 forward passes of compute (2.5 for the MLP measured here), while forward mode or finite differences cost one pass per parameter. The price is memory for stored activations, which checkpointing trades back for recomputation."]
question: "In your scalar autograd engine, `y = x * x` with `x = 3` gives `x.grad == 3` after `y.backward()` instead of 6. The multiply's backward is `self.grad = other.data * out.grad; other.grad = self.data * out.grad`. Explain the bug through the graph structure and why the same bug would also break a recurrent network with shared weights. Then explain how you would have caught it with a test, and why a PyTorch training loop that forgets `optimizer.zero_grad()` shows the opposite problem."
answer: "In `x * x` both inputs of the multiply node are the same object, so the graph has two edges from `x` into that node: `x` has a fan-out of two. By the multivariable chain rule its adjoint is the sum of the contributions along both edges: $\\partial y/\\partial x = x + x = 6$. With assignment, the second line overwrites the first: the first sets `x.grad = 3`, the second sets it to 3 again instead of adding, so one path's contribution is lost. Using `+=` gives 3 + 3 = 6.\n\nA recurrent network applies the same weight matrix at every time step, so in the unrolled graph that weight has one consumer per step and its gradient must be the sum over all steps (backprop through time). With assignment, only the contribution written last would survive, so the weights would be trained as if the sequence had length 1, silently and without any error.\n\nA gradient check catches it immediately: the central difference $((3.001)^2 - (2.999)^2)/0.002 = 6$ in float64 disagrees with the analytic 3 by a relative error of 0.5, far above the roughly 1e-7 you expect from a correct backward. Including a test case where a value is reused (a square, a shared weight, a residual connection) is essential, because graphs without fan-out would pass even with the bug.\n\nForgetting `zero_grad()` is the opposite mistake: accumulation that should be reset is not. PyTorch adds every `backward()` into the existing `.grad`, which is correct within one step (that is how fan-out is summed and how gradient accumulation over micro-batches works) but means that across steps the gradient becomes the sum of all past gradients. The optimiser then keeps stepping in stale directions, behaving like momentum with no decay, and the loss oscillates or diverges instead of converging. The general rule: any node with fan-out greater than one needs accumulation within a backward pass, and accumulated gradients must be zeroed before the next one."
sources: [{"title": "Karpathy: micrograd", "url": "https://github.com/karpathy/micrograd", "note": "A complete scalar autograd engine and neural-net library in about 150 lines; read engine.py."}, {"title": "CS231n: Backpropagation, Intuitions", "url": "https://cs231n.github.io/optimization-2/", "note": "Gates as local gradient rules, patterns in backward flow, staged computation."}, {"title": "Olah: Calculus on Computational Graphs", "url": "https://colah.github.io/posts/2015-08-Backprop/", "note": "Why reverse mode beats forward mode, with path-sum diagrams."}, {"title": "Baydin et al.: Automatic Differentiation in Machine Learning: a Survey", "url": "https://arxiv.org/abs/1502.05767", "note": "Forward vs reverse mode, cost analysis, and how real AD systems are built."}]
practice: ["run-reverse-mode-autodiff", "backpropagation-through-computation-graph", "accumulate-shared-path-gradients", "topologically-sort-computation-dag", "gradient-check-autodiff-graph"]
tags: ["backprop", "autodiff", "computational graph"]
tools: ["Python"]
---

## What you will learn

- What a **computational graph** is, and how any network (or any formula) can be drawn as one.
- The two passes of backpropagation: a forward pass that computes and stores values, and a backward pass that sends gradients from the loss back to every input and weight.
- The local rules (add, multiply, tanh, matrix multiply) that are enough to differentiate a whole MLP.
- Why gradients must be **added up** when a value is used more than once, and why PyTorch makes you call `zero_grad()`.
- How to test a hand-written backward pass with finite differences, and what step size to use.
- Why reverse mode is the right choice for training: it costs a few forward passes no matter how many weights you have, and pays for that with memory.

*Before you start:* you need the MLP from lesson 01.01, the activation derivatives from 01.02 (for example $\tanh'(z) = 1 - \tanh^2(z)$), and the loss functions of 01.03. You should know the one-variable chain rule: if $L = f(g(x))$ then $\frac{dL}{dx} = f'(g(x))\,g'(x)$.

## From a loss to a gradient

Lesson 01.03 ended with a single number, the loss, that says how wrong the network is on a batch. Training needs more than that number. For each of possibly millions of weights it needs to know: if I nudge this weight up a little, does the loss go up or down, and how fast? That list of slopes is the **gradient**, and the optimiser steps every weight against its own slope.

**Backpropagation** (backprop) is the algorithm that computes this gradient. It is the chain rule, applied to a graph of simple operations, in an order that never computes the same thing twice. PyTorch's `loss.backward()` runs exactly this algorithm; after this lesson you will be able to write a small version of it yourself.

## The idea in one picture

Take a tiny formula with four inputs: $L = (a \cdot b + c) \cdot f$, with $a = 3$, $b = -2$, $c = 5$, $f = -4$. Break it into one operation per step and give each intermediate result a name:

| Step | Operation | Value
| --- | --- | --- |
| 1 | $d = a \cdot b$ | $3 \times (-2) = -6$
| 2 | $e = d + c$ | $-6 + 5 = -1$
| 3 | $L = e \cdot f$ | $(-1) \times (-4) = 4$

Drawn as boxes and arrows, this is a **computational graph**: each box (a **node**) holds a value, and an arrow means "this value is an input to that operation". Inputs and weights with no incoming arrow are the **leaves**; the final scalar, here $L$, is the **root**.

~~~dg-fig 01.04-graph.svg
The graph for L = (a·b + c)·f. Blue numbers come from the forward pass (left to right); orange numbers are the gradients dL/d(node) from the backward pass (right to left). The root's gradient is 1 by definition.
~~~

Each orange number answers one question: how much does $L$ change per unit change in that node? For example, the gradient of $a$ is 8, so raising $a$ from 3 to 3.001 should raise $L$ by about $0.008$. Check it: $d = -6.002$, $e = -1.002$, $L = 4.008$. It does.

Every orange number was computed from just the orange number after it and one simple fact about one operation. Nobody wrote a formula for $\partial L / \partial a$ as a whole. That locality is the whole trick.

## Step 1: break the computation into a graph

Any computation built from a fixed menu of primitive operations (add, multiply, exp, tanh, matrix multiply, sum, and so on) can be drawn this way. The graph is a **DAG**, a directed acyclic graph: arrows have a direction and you can never follow them in a loop, because a value cannot depend on itself.

For a neural network the leaves are the input batch, the labels and every weight and bias; the internal nodes are the layer operations from 01.01 (`X @ W1`, `+ b1`, `tanh`, ...); the root is the scalar loss.

A **topological order** is any ordering of the nodes in which every node comes after all of its inputs. Reading the table above top to bottom is one. The forward pass visits nodes in topological order; the backward pass visits them in the reverse of it.

## Step 2: the forward pass computes and remembers

Go through the nodes in topological order and compute each value from its inputs. This is just running the network. The one extra duty is to **keep** the intermediate values (the blue numbers), because the backward pass will need them. In the figure, computing the gradient of $a$ needs the value of $b$, and computing the gradient of $e$ needs the value of $f$.

In a network these stored intermediates are the layers' **activations**; their memory cost is measured at the end.

## Step 3: one local rule per operation

Write $\bar v$ for $\partial L / \partial v$, the gradient of the final loss with respect to node $v$. It is often called the **adjoint** of $v$. Each primitive operation needs exactly one rule: given the adjoint of its output (the **upstream gradient**), produce a contribution to the adjoint of each input. The rule only uses the operation's own inputs and output, so it is **local**:

| Operation | Rule for the inputs' adjoints | In words
| --- | --- | --- |
| $c = a + b$ | $\bar a \mathrel{+}= \bar c$, $\bar b \mathrel{+}= \bar c$ | addition passes the gradient through unchanged
| $c = a \cdot b$ | $\bar a \mathrel{+}= b\,\bar c$, $\bar b \mathrel{+}= a\,\bar c$ | multiplication hands each input the *other* input's value
| $c = \tanh(a)$ | $\bar a \mathrel{+}= (1 - c^2)\,\bar c$ | scale by the local slope of tanh
| $C = AB$ (matrices) | $\bar A \mathrel{+}= \bar C B^\top$, $\bar B \mathrel{+}= A^\top \bar C$ | the same "swap the other input" idea, with transposes to make shapes fit

Each rule is the chain rule in miniature: (gradient arriving from above) × (local derivative of this operation). The `+=` instead of `=` is deliberate and is explained in step 5.

The matrix rule is easiest to remember by shapes. If $A$ is $m{\times}n$ and $B$ is $n{\times}p$, then $\bar C$ is $m{\times}p$ and $\bar A$ must be $m{\times}n$; the only way to combine $\bar C$ and $B$ into that shape is $\bar C B^\top$.

## Step 4: the backward pass, in reverse order

Now run the graph backwards:

1. Set the root's adjoint to 1, since $\partial L / \partial L = 1$. Set every other adjoint to 0.
2. Visit nodes in reverse topological order. At each node, apply its local rule: multiply its (now complete) adjoint by the local derivatives and add the results into its inputs' adjoints.
3. When you reach the leaves, their adjoints are the gradient you wanted.

Why reverse order? A node's adjoint is finished only once *every* node that uses it has contributed, and reverse topological order visits all consumers first.

<div class="ex">
<div class="lab">Worked example</div>

**backprop by hand on the picture's graph**

Forward (from the table): $d = -6$, $e = -1$, $L = 4$.

Start: $\bar L = 1$.

Node $L = e \cdot f$ (multiply): $\bar e = f \cdot \bar L = -4 \times 1 = -4$, and $\bar f = e \cdot \bar L = -1 \times 1 = -1$.

Node $e = d + c$ (add): passes $\bar e$ through, so $\bar d = -4$ and $\bar c = -4$.

Node $d = a \cdot b$ (multiply): $\bar a = b \cdot \bar d = (-2)(-4) = 8$, and $\bar b = a \cdot \bar d = 3 \times (-4) = -12$.

Check $b$ numerically: $b = -2.001$ gives $d = -6.003$, $e = -1.003$, $L = 4.012$, so the slope is $0.012 / (-0.001) = -12$. Matches.

</div>

## Step 5: when a value is used twice, add the paths

Real networks reuse values constantly: the input feeds every first-layer neuron, a recurrent network reuses one weight at every time step, a residual block sends $x$ through a layer and around it. When a node has more than one consumer (its **fan-out** is above one), the multivariable chain rule says its adjoint is the **sum** of the contributions along every path.

The smallest case is $y = x \cdot x$ with $x = 3$. The multiply rule sends $x \cdot \bar y = 3$ to its first input and $3$ to its second input. Both inputs are the same node, so $\bar x = 3 + 3 = 6$, which is the familiar $\frac{d}{dx}x^2 = 2x = 6$. If the rule used `=` instead of `+=`, the second write would overwrite the first and you would get 3: half the true gradient, with no error message.

The same habit of accumulating has a practical consequence in PyTorch: `.grad` buffers are *added to* by every `backward()` call, never overwritten. That enables gradient accumulation over small batches (lesson 02.04), but it means you must reset them before each new step with `optimizer.zero_grad()`.

~~~dg-fig 01.04-zerograd.svg
Linear regression trained with SGD (learning rate 0.05). With zero_grad the loss falls to 0.011. Without it, each step adds the new gradient to all the old ones, so stale gradients keep pushing the weights and the loss bounces between about 1 and 4 forever.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one tanh neuron with a squared-error loss**

Weight $w = 0.5$, input $x = 2$, bias $b = -0.5$, target $y = 0.8$. The graph is $z = w x + b$, $t = \tanh(z)$, $L = (t - y)^2$.

Forward: $z = 0.5 \times 2 - 0.5 = 0.5$. $t = \tanh(0.5) = 0.4621$. $L = (0.4621 - 0.8)^2 = (-0.3379)^2 = 0.1142$.

Backward through the square: $\bar t = 2(t - y) = 2 \times (-0.3379) = -0.6758$.

Through tanh: the local slope is $1 - t^2 = 1 - 0.2135 = 0.7864$, so $\bar z = -0.6758 \times 0.7864 = -0.5315$.

Through the add: $\bar b = \bar z = -0.5315$, and the product $wx$ also receives $-0.5315$.

Through the multiply: $\bar w = x \cdot \bar z = 2 \times (-0.5315) = -1.0629$ and $\bar x = w \cdot \bar z = 0.5 \times (-0.5315) = -0.2657$.

Reading: $\bar w$ is negative, so increasing $w$ lowers the loss. That makes sense, since the neuron outputs 0.46 and wants 0.8. A gradient descent step with learning rate 0.1 would set $w \leftarrow 0.5 - 0.1 \times (-1.0629) = 0.606$.

</div>

## Backprop in code

The whole algorithm fits in a few dozen lines. Each value remembers its parents and a small function (a closure) that applies its local rule. `backward()` sorts the graph topologically with a depth-first search and then calls those functions in reverse:

```python
import math

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = data, 0.0
        self._parents, self._back = parents, lambda: None
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def back(): self.grad += out.grad; o.grad += out.grad
        out._back = back; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def back(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._back = back; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def back(): self.grad += (1 - t * t) * out.grad
        out._back = back; return out
    def backward(self):
        order, seen = [], set()
        def visit(v):                       # depth-first search = topological sort
            if v not in seen:
                seen.add(v)
                for p in v._parents: visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order): v._back()

a, b, c, f = Value(3.0), Value(-2.0), Value(5.0), Value(-4.0)
L = (a * b + c) * f
L.backward()
print(L.data, a.grad, b.grad, c.grad, f.grad)   # 4.0 8.0 -12.0 -4.0 -1.0
x = Value(3.0); (x * x).backward(); print(x.grad)   # 6.0
```

Karpathy's micrograd is this design plus a few more operations. PyTorch is the same idea over tensors: each operation records a backward node as it runs (the graph is rebuilt on every forward pass, which is why Python `if` and `for` inside a model just work), and each rule is a compiled kernel.

Real implementations apply the rules to whole matrices. Here is a two-layer network, backpropagated by hand in NumPy with the matrix rule, then checked against finite differences and PyTorch:

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(8, 3)); y = rng.normal(size=(8, 1))          # batch of 8, 3 features
W1 = rng.normal(size=(3, 4)) * 0.5; W2 = rng.normal(size=(4, 1)) * 0.5

def forward(W1, W2):
    Z = X @ W1; H = np.tanh(Z); P = H @ W2                         # P: predictions (8, 1)
    return np.mean((P - y) ** 2), (Z, H, P)

L, (Z, H, P) = forward(W1, W2)
dP = 2 * (P - y) / len(X)          # dL/dP from the mean squared error
dW2 = H.T @ dP                     # matmul rule: B-bar = A^T C-bar
dH = dP @ W2.T                     # matmul rule: A-bar = C-bar B^T
dZ = dH * (1 - H ** 2)             # tanh rule (uses the stored H)
dW1 = X.T @ dZ

num = np.zeros_like(W1); h = 1e-5  # central finite differences, entry by entry
for i in np.ndindex(W1.shape):
    Wp, Wm = W1.copy(), W1.copy(); Wp[i] += h; Wm[i] -= h
    num[i] = (forward(Wp, W2)[0] - forward(Wm, W2)[0]) / (2 * h)
print(f"loss {L:.4f}  max relative error {np.abs(num - dW1).max() / np.abs(num).max():.1e}")
# loss 0.9309  max relative error 5.0e-11

import torch
tW1 = torch.tensor(W1, requires_grad=True); tW2 = torch.tensor(W2, requires_grad=True)
((torch.tanh(torch.tensor(X) @ tW1) @ tW2 - torch.tensor(y)) ** 2).mean().backward()
print(np.allclose(tW1.grad.numpy(), dW1), np.allclose(tW2.grad.numpy(), dW2))   # True True
```

Notice that the backward lines mirror the forward lines in reverse, and that `dZ` reuses `H`, the stored activation. Every backward pass you will ever write has this shape.

## Checking a backward pass

A wrong gradient rarely crashes anything; the network just trains worse. The standard one is the **gradient check**: compare each analytic gradient with a numerical slope,

$$\frac{\partial L}{\partial w} \approx \frac{L(w + h) - L(w - h)}{2h},$$

the **central difference**, where $h$ is a small step. Report the **relative error** $|g_{\text{analytic}} - g_{\text{numeric}}| / |g_{\text{numeric}}|$ so the test does not depend on the scale of the loss.

The step size matters: too large and curvature spoils the estimate; too small and subtracting two nearly equal floats lets rounding error take over.

~~~dg-fig 01.04-gradcheck.svg
Relative error of finite-difference estimates for the tanh neuron above, as the step h shrinks. Central differences in float64 reach about 1e-11 near h = 1e-6; one-sided differences are far worse; in float32 the best you can get is about 2e-6, near h = 1e-3.
~~~

Practical rules that follow from the figure:

- Use central differences in **float64**. In float64, a relative error around $10^{-7}$ or smaller means the backward is right; around $10^{-2}$ or larger means a bug.
- Do not gradient-check in float32: even a perfect backward shows errors near $10^{-6}$ at the best step and much worse at the usual $h = 10^{-5}$ (about $2 \times 10^{-3}$ for this neuron), which looks like a bug.
- Turn off dropout and other randomness, or the two sides of the difference see different functions.
- In PyTorch, `torch.autograd.gradcheck(fn, inputs)` does all of this for a custom operation; it expects float64 inputs.

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| A reused value gets only part of its gradient (for example `x * x` gives 3 instead of 6) | backward rule assigns with `=` instead of accumulating with `+=` | always accumulate into adjoints
| Loss oscillates or diverges although the code "looks right" | `zero_grad()` missing, so gradients from earlier steps pile up | call `optimizer.zero_grad()` once per step, before `backward()`
| `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` | an in-place op such as `x.mul_(2)` overwrote a value the backward pass had stored | use the out-of-place version, or clone first
| `RuntimeError: grad can be implicitly created only for scalar outputs` | called `backward()` on a tensor that is not a single number | reduce to a scalar loss first (`.mean()`), or pass an explicit upstream gradient
| A weight's `.grad` stays `None` | the graph was cut: `.item()`, `.numpy()`, `.detach()` or a `torch.no_grad()` block sat between the weight and the loss | keep the whole path in tensor operations with gradients on
| Gradient check fails by about 1e-3 on correct code | checking in float32, or with dropout active | float64, fixed seeds, evaluation mode
| Shape error in a hand-written matmul backward | transposes in the wrong place | derive $\bar A = \bar C B^\top$ by matching shapes

## Going deeper: why reverse mode, and what it costs

**Cost.** There are two ways to push the chain rule through a graph. **Forward mode** starts at one input and carries "how much does every node change if this input moves" forward through the graph. One forward-mode pass gives the derivative with respect to *one* input direction; this is a **Jacobian-vector product (JVP)**. **Reverse mode** starts at one output and carries adjoints backwards; one pass gives the derivative of *one* output with respect to *every* input, a **vector-Jacobian product (VJP)**. Backprop is reverse mode.

Training has millions of inputs (the weights) and one output (the loss), so the choice is lopsided. Counting floating-point operations with PyTorch's `FlopCounterMode` on a three-layer tanh MLP (width 256, batch 64, 134,154 parameters): forward plus backward cost 2.5 forward passes (under 3 because the first layer needs no gradient for the input data), and one JVP cost 3.0. Forward mode needs one JVP per parameter.

~~~dg-fig 01.04-cost.svg
Cost of one full gradient, measured in forward passes. Finite differences and forward mode grow in proportion to the number of parameters; reverse mode stays at about 2.5 forward passes (the FLOP ratio measured on this MLP) whether there are a hundred parameters or seven billion.
~~~

Forward mode wins in the opposite situation, few inputs and many outputs. JAX exposes both as `jax.jvp` and `jax.vjp`; PyTorch has `torch.func.jvp` and `torch.func.vjp`.

**Memory.** Reverse mode's price is that the backward rules need forward values (the other input of every multiply, the output of every tanh), so they stay in memory until used. Measuring the tensors PyTorch saves for backward (with `torch.autograd.graph.saved_tensors_hooks`) on a stack of 256-wide tanh layers with a batch of 64 gives 64 KB per layer: 0.31 MB for 4 layers, 4.06 MB for 64. For large models, these stored activations, not the weights, are often what fills the GPU.

~~~dg-fig 01.04-memory.svg
Stored activations grow in a straight line with depth (measured). Checkpointing with √depth segments keeps only the segment boundaries plus one recomputed segment at a time (estimated from the measured 64 KB per layer), so memory grows like √depth: about 1 MB instead of 4 MB at 64 layers.
~~~

**Activation checkpointing** (`torch.utils.checkpoint`) trades compute for memory: it stores the activations only at segment boundaries, then re-runs each segment's forward during backward to regenerate what is needed. With $\sqrt{N}$ segments of $\sqrt{N}$ layers, memory falls from order $N$ to order $\sqrt{N}$, for the price of roughly one extra forward pass (about 40% more compute when backward costs 2.5 forwards, as above).

**Edges and extensions.**

- **Non-differentiable points.** ReLU has no derivative at 0. Frameworks simply pick a value: PyTorch's ReLU backward uses 0 there.
- **Higher derivatives.** The backward pass is itself a computation made of primitive operations, so it can be recorded as a graph and differentiated again. In PyTorch, `torch.autograd.grad(y, x, create_graph=True)` does this; for $y = x^3$ at $x = 3$ it gives 27, and differentiating that again gives 18.
- **Unrolled loops.** A recurrent network over $T$ steps is a graph with $T$ copies of the cell sharing one weight. Backprop through it ("backprop through time") is ordinary backprop, and the shared weight's gradient is the sum over the $T$ copies, exactly the fan-out rule of step 5.

## Key terms

| Term | Meaning
| --- | --- |
| Computational graph | A DAG whose nodes are primitive operations and whose edges carry values from one operation to the next
| Forward pass | Evaluating the graph in topological order, storing intermediate values
| Backward pass | Visiting nodes in reverse topological order, turning each output gradient into input gradients
| Adjoint $\bar v$ | $\partial L / \partial v$: how much the final loss changes per unit change in node $v$
| Local rule | The derivative rule of one operation, needing only its own inputs and output
| Fan-out accumulation | A value used several times gets the sum of the gradients from all its uses
| Gradient check | Comparing analytic gradients with central finite differences in float64
| Activation checkpointing | Storing fewer activations and recomputing the rest during backward to save memory

<div class="take">
<div class="lab">Key takeaway</div>

backprop runs the graph forward while storing values, then walks it in reverse applying one local rule per operation and summing over every path a value takes. That gives the gradient for every weight at about 2-3 forward passes of compute, paid for with the memory of stored activations; test any hand-written backward with a float64 central-difference gradient check.

</div>
