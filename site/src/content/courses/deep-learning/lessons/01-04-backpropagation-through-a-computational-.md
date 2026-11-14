---
id: "01.04"
course: "deep-learning"
module: "01"
slug: "backpropagation-through-a-computational-graph"
title: "Backpropagation through a computational graph"
keyIdeas: ["Any computation built from simple operations forms a directed acyclic graph; each node only needs its local derivative to take part in backprop.", "Reverse-mode autodiff runs the graph forward, then visits nodes in reverse topological order, multiplying the upstream gradient by each local derivative (the chain rule).", "When a value feeds several consumers its gradient is the **sum** over all paths, so gradients must be accumulated with `+=`, and reset between steps.", "Reverse mode gives the gradient of one scalar loss with respect to millions of parameters for about 2-3× the cost of the forward pass, at the price of storing intermediate activations."]
question: "In your scalar autograd engine, `y = x * x` with `x = 3` gives `x.grad == 3` after `y.backward()` instead of 6. The multiply's backward is `self.grad = other.data * out.grad; other.grad = self.data * out.grad`. Explain the bug through the graph structure and why the same bug would also break a recurrent network with shared weights."
answer: "In `x * x` both inputs of the multiply node are the same object, so the node has two edges into `x`. The true derivative sums both paths: $\\partial y/\\partial x = x + x = 6$. With assignment, the second line overwrites the first: the first sets `x.grad = 3`, the second sets it to 3 again instead of adding, so one path's contribution is lost. Using `+=` gives 3 + 3 = 6. A recurrent network reuses the same weight matrix at every time step, so in the unrolled graph that weight has one consumer per step and its gradient must be the sum over all steps (backprop through time). With assignment, only the last-processed step's contribution would survive, so the weights would be trained as if the sequence had length 1. The general rule: any node with fan-out greater than one needs accumulation, and accumulated gradients must be zeroed before the next backward."
sources: [{"title": "Karpathy: micrograd", "url": "https://github.com/karpathy/micrograd", "note": "A complete scalar autograd engine and neural-net library in about 150 lines; read engine.py."}, {"title": "CS231n: Backpropagation, Intuitions", "url": "https://cs231n.github.io/optimization-2/", "note": "Gates as local gradient rules, patterns in backward flow, staged computation."}, {"title": "Olah: Calculus on Computational Graphs", "url": "https://colah.github.io/posts/2015-08-Backprop/", "note": "Why reverse mode beats forward mode, with path-sum diagrams."}, {"title": "Baydin et al.: Automatic Differentiation in Machine Learning: a Survey", "url": "https://arxiv.org/abs/1502.05767", "note": "Forward vs reverse mode, cost analysis, and how real AD systems are built."}]
practice: ["run-reverse-mode-autodiff", "backpropagation-through-computation-graph", "accumulate-shared-path-gradients", "topologically-sort-computation-dag", "gradient-check-autodiff-graph"]
tags: ["backprop", "autodiff", "computational graph"]
tools: ["Python"]
---

## From chain rule to algorithm

You know the chain rule: if $L = f(g(x))$ then $\frac{dL}{dx} = f'(g(x))\,g'(x)$. Backpropagation is the chain rule applied systematically to a graph with millions of nodes, in an order that reuses work. ML Maths and AI Research's chain-rule lessons cover the calculus; here the point is the **algorithm** and the engineering around it, the thing PyTorch's autograd does every step.

## The computational graph

Write any computation as a DAG: leaves are inputs and parameters, internal nodes are primitive operations (add, multiply, exp, tanh, matmul), and the root is the scalar loss. For each primitive you only need one rule: given the gradient of $L$ with respect to the node's output, produce the gradient with respect to each input. That rule needs only local information:

- $c = a + b$: $\bar a \mathrel{+}= \bar c$, $\bar b \mathrel{+}= \bar c$ (addition routes gradient unchanged).
- $c = a \cdot b$: $\bar a \mathrel{+}= b\,\bar c$, $\bar b \mathrel{+}= a\,\bar c$ (multiplication swaps the inputs).
- $c = \tanh(a)$: $\bar a \mathrel{+}= (1 - c^2)\,\bar c$.
- $C = AB$ (matrices): $\bar A \mathrel{+}= \bar C B^\top$, $\bar B \mathrel{+}= A^\top \bar C$.

Here $\bar v$ means $\partial L / \partial v$, the "adjoint". The global derivative falls out of composing local rules.

~~~dg-flow Forward computes values left to right; backward sends adjoints right to left, each node applying only its local rule.
[Forward] a, b -(mul)-> d = a·b -(add c)-> e = d + c -(mul f)-> *L = e·f
[Backward] L̄ = 1 -(f̄ = e, ē = f)-> ē -(c̄ = ē, d̄ = ē)-> d̄ -(ā = b·d̄, b̄ = a·d̄)-> *Leaf grads
~~~

## Why reverse order, and why accumulate

A node's adjoint is complete only after **every** consumer has sent its contribution. Visiting nodes in reverse topological order guarantees that. If a value $x$ feeds two operations, its adjoint is the sum of the two path contributions, by the multivariate chain rule. That is why every rule above uses `+=`. Two classic bugs follow:

- Using `=` instead of `+=`: a reused node (a weight shared across time steps, or $x$ in $x \cdot x$) silently loses all but one path.
- Forgetting to reset gradients between steps: PyTorch accumulates into `.grad` by design, which is why every loop calls `optimizer.zero_grad()`. The same behaviour is what makes gradient accumulation across micro-batches work (02.04).

## A scalar autograd engine

The whole idea fits in a few dozen lines. Each value remembers its parents and a closure that applies its local rule:

```python
import math

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad, self._parents, self._back = data, 0.0, parents, lambda: None
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o); out = Value(self.data + o.data, (self, o))
        def back(): self.grad += out.grad; o.grad += out.grad
        out._back = back; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o); out = Value(self.data * o.data, (self, o))
        def back(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._back = back; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def back(): self.grad += (1 - t * t) * out.grad
        out._back = back; return out
    def backward(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v); [visit(p) for p in v._parents]; order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v._back()
```

Karpathy's micrograd is this design plus a few more operations and a tiny neural-network library on top; PyTorch is the same idea over tensors, with each primitive's backward written as a fused kernel.

<div class="ex">
<div class="lab">Worked example</div>

**backprop by hand**

Let $a = 3$, $b = -2$, $c = 5$, $f = -4$. Forward: $d = ab = -6$, $e = d + c = -1$, $L = ef = 4$.

Backward: $\bar L = 1$. Multiply node: $\bar e = f = -4$, $\bar f = e = -1$. Add node routes unchanged: $\bar d = \bar c = -4$.

Multiply node: $\bar a = b\,\bar d = (-2)(-4) = 8$, $\bar b = a\,\bar d = 3 \cdot (-4) = -12$.

Check one numerically: raising $a$ to 3.001 gives $d = -6.002$, $e = -1.002$, $L = 4.008$, a slope of 8. Finite differences like this are the standard way to test any hand-written backward.

</div>

## Cost, memory, and why reverse mode

For a function $\mathbb{R}^n \to \mathbb{R}$ (millions of parameters, one loss), reverse mode computes all $n$ partial derivatives in **one** backward pass costing a small constant times the forward pass, in practice 2-3× for matmul-heavy nets. Forward-mode autodiff would need $n$ passes, one per input direction. Forward mode wins only when outputs outnumber inputs. In Jacobian terms, reverse mode computes vector-Jacobian products, forward mode Jacobian-vector products.

The price is **memory**: the backward rules need forward values ($b$ for $\bar a$, $t$ for tanh), so every intermediate activation is stored until its backward runs. For large models activations, not weights, often dominate memory. Activation checkpointing trades compute for memory by storing only some activations and recomputing the rest during backward.

<div class="take">
<div class="lab">Key takeaway</div>

backprop is reverse topological traversal applying local derivative rules and summing over paths; it gives every parameter's gradient for a few forward passes' compute, paid for with stored activations.

</div>
