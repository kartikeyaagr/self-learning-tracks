---
id: "16.05"
course: "ml-maths-course"
module: "16"
slug: "computational-graphs-and-backpropagation"
title: "Computational graphs and backpropagation"
keyIdeas: ["A computation is a directed acyclic graph: nodes are operations or values, edges carry intermediate results; the forward pass visits nodes in topological order.", "Reverse-mode differentiation visits nodes in reverse topological order, multiplying local derivatives and summing contributions where a value feeds several consumers.", "For a scalar loss of many parameters, reverse mode costs a small constant multiple of the forward pass, which is why backpropagation scales to billions of parameters.", "Memory, not compute, is the usual limit: the backward pass needs forward activations, and checkpointing trades recomputation for memory."]
question: "Let $a = x \\cdot y$, $b = a + x$, and $f = b^2$, with $x = 2$ and $y = 3$. (a) Draw the computation graph and give a topological order. (b) Run the backward pass to find $\\partial f/\\partial x$ and $\\partial f/\\partial y$. (c) Which node needs its gradient contributions summed, and what goes wrong if you overwrite instead of adding?"
answer: "(a) Nodes $x$ and $y$ feed $a$; $a$ and $x$ feed $b$; $b$ feeds $f$. One topological order is $x, y, a, b, f$. Forward: $a = 6$, $b = 8$, $f = 64$. (b) $\\bar{f} = 1$; $\\bar{b} = 2b = 16$; $\\bar{a} = \\bar{b} \\cdot 1 = 16$; $\\bar{y} = \\bar{a}\\cdot x = 32$; $\\bar{x} = \\bar{a}\\cdot y + \\bar{b}\\cdot 1 = 48 + 16 = 64$. Check: $f = (xy + x)^2$, so $\\partial f/\\partial x = 2(xy + x)(y + 1) = 2 \\cdot 8 \\cdot 4 = 64$ and $\\partial f/\\partial y = 2(xy + x)\\,x = 32$. (c) $x$ feeds both $a$ and $b$, so its gradient is the sum of two paths. Overwriting would keep only one path and give 48 or 16 instead of 64."
sources: [{"title": "Wikipedia, Automatic differentiation", "url": "https://en.wikipedia.org/wiki/Automatic_differentiation", "note": "Forward and reverse modes and their costs."}, {"title": "Wikipedia, Topological sorting", "url": "https://en.wikipedia.org/wiki/Topological_sorting", "note": "Ordering nodes of a directed acyclic graph."}, {"title": "Chen et al., Training Deep Nets with Sublinear Memory Cost (2016)", "url": "https://arxiv.org/abs/1604.06174", "note": "Gradient checkpointing."}]
practice: ["topologically-sort-computation-dag", "build-scalar-expression-graph", "run-reverse-mode-autodiff", "accumulate-shared-path-gradients"]
tags: ["computational graph", "topological order", "reverse-mode autodiff", "gradient accumulation"]
tools: ["Python", "PyTorch"]
---

## Where this lesson sits

ML Maths 09.19 introduced the chain rule on computation graphs and 09.38 worked through backpropagation for a linear layer. This lesson takes the graph view: what properties of the graph make backpropagation correct and efficient, and how they connect to the rest of this module.

## Computations as directed acyclic graphs

Any program that computes a loss from inputs and parameters, without loops that depend on values, can be written as a directed acyclic graph (DAG). Leaves are inputs and parameters; internal nodes are operations; edges carry values. Because there are no cycles, the nodes have a topological order in which every node comes after all its inputs. The forward pass evaluates nodes in that order. (Loops and recursion are unrolled into a larger DAG for the specific input, which is what dynamic frameworks record.)

Topological sorting takes linear time: repeatedly emit a node with no unprocessed inputs (Kahn's algorithm), or run a depth-first search and reverse the finishing order.

## The backward pass

Write $\bar{v} = \partial f/\partial v$ for the gradient of the final scalar $f$ with respect to node $v$ (the "adjoint"). Start with $\bar{f} = 1$ and visit nodes in reverse topological order. Each node $u$ that consumes $v$ contributes $\bar{u}\,\partial u/\partial v$, and contributions add:
$\bar{v} = \sum_{u \in \text{consumers}(v)} \bar{u}\,\frac{\partial u}{\partial v}$.
Reverse topological order guarantees that every consumer's adjoint is final before it is used.

~~~dg-flow Forward and backward passes on a DAG.
Inputs and parameters -(topological order)-> Operations -(store activations)-> *Scalar loss -(reverse topological order: multiply local derivatives, sum over consumers)-> Gradients for every parameter
~~~

## Why reverse mode

For $f: \mathbb{R}^n \to \mathbb{R}$, forward-mode differentiation propagates one directional derivative per pass, so all $n$ partial derivatives need $n$ passes. Reverse mode propagates the adjoint of the single output backward and yields all $n$ partial derivatives in one pass, at a cost of a small constant (typically 2 to 3) times the forward pass. In matrix language, reverse mode computes vector-Jacobian products, forward mode computes Jacobian-vector products. Training has one scalar loss and many parameters, so reverse mode wins by a factor of $n$.

<div class="ex">
<div class="lab">Worked example</div>

**the gradient that was half its true size**

A hand-written autodiff engine passed unit tests on chains of operations but gave wrong gradients for a model with weight tying, where one embedding matrix was used both for input lookups and for output logits. The engine assigned gradients with "=" instead of "+=", so whichever use was processed last overwrote the other. Gradient checks with finite differences flagged the tied matrix immediately; switching to accumulation fixed it.

</div>

## Memory

The backward pass needs values from the forward pass (for example, the input to a multiplication to compute its derivative). Storing all activations costs memory proportional to the depth times the batch size times the width, often more than the parameters themselves. Gradient checkpointing stores only some activations and recomputes the rest during the backward pass; storing every $\sqrt{n}$-th layer of an $n$-layer network gives $O(\sqrt{n})$ memory for about one extra forward pass.

## Graph structure and efficiency

- Shared subexpressions: a value used by several consumers is computed once and its adjoint summed, which is where reverse mode saves work compared with differentiating each path separately.
- Fan-in and fan-out determine where gradients accumulate and where parallelism is possible.
- Compilers fuse chains of element-wise nodes into one kernel, reducing memory traffic; the DAG makes these rewrites safe.
- Higher-order derivatives differentiate the backward graph itself; Hessian-vector products cost a few passes, not $n$.

## Links to the rest of the module

The computation graph is a different kind of graph from the data graphs elsewhere in this module: its nodes are operations, not entities. But the same ideas recur. Message passing in graph neural networks (lesson 16.06) is a forward pass over a data graph, and its backward pass runs over the computation graph built from it, with gradient flowing back along every edge used for a message.

<div class="take">
<div class="lab">Key takeaway</div>

backpropagation is reverse-topological traversal of a computation DAG, multiplying local derivatives and summing over consumers; one backward pass gives every parameter's gradient for a small constant cost, and memory for activations is the main limit.

</div>
