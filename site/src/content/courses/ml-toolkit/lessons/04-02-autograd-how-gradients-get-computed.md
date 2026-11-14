---
id: "04.02"
course: "ml-toolkit"
module: "04"
slug: "autograd-how-gradients-get-computed"
title: "Autograd: how gradients get computed"
keyIdeas: ["Autograd records a graph *while* your forward code runs: every op on a tensor with `requires_grad=True` produces an output whose `grad_fn` points back at its inputs. The graph is rebuilt on every forward pass, so Python control flow just works.", "`loss.backward()` walks that graph in reverse, applying each op's vector-Jacobian product, and **adds** the result into `.grad` of each leaf. Nothing is overwritten, which is why you must zero gradients between steps and why gradient accumulation is free.", "After `backward()` the graph's saved tensors are freed. A second backward through the same graph needs `retain_graph=True`; higher-order gradients need `create_graph=True`.", "`torch.no_grad()` and `torch.inference_mode()` stop recording (for evaluation and optimiser updates); `.detach()` cuts one tensor out of the graph while sharing its storage; `requires_grad_(False)` freezes parameters."]
question: "You train with gradient accumulation over 8 micro-batches but write `loss = loss_fn(out, y); loss.backward()` without dividing by 8, and you also call `opt.zero_grad()` inside the micro-batch loop instead of before it. Describe precisely what gradient each optimiser step uses, and what each bug alone would do to training."
answer: "With `zero_grad()` inside the loop, each micro-batch's gradients are wiped before the next one is added, so when `opt.step()` runs, `.grad` holds only the last micro-batch's gradient. Training then behaves like batch size B/8 with 7/8 of the data's gradient signal thrown away, plus the noise of a small batch. Without that bug, forgetting the division by 8 makes the step use the sum rather than the mean of 8 micro-batch gradients: the direction is right but the magnitude is 8 times too large, equivalent to multiplying the learning rate by 8 for SGD. Adam largely normalises the scale away (it divides by the root of the second moment), though epsilon and weight decay interactions shift slightly, while gradient clipping thresholds become wrong by a factor of 8. Both fixes: zero once before the loop, and divide each micro-batch loss by the number of micro-batches (or by total tokens)."
sources: [{"title": "PyTorch docs: Autograd mechanics", "url": "https://pytorch.org/docs/stable/notes/autograd.html", "note": "How the graph is recorded, saved tensors, in-place correctness checks and grad modes."}, {"title": "PyTorch tutorial: A gentle introduction to torch.autograd", "url": "https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html", "note": "Vector-Jacobian products and the DAG, with small runnable examples."}, {"title": "PyTorch docs: torch.autograd", "url": "https://pytorch.org/docs/stable/autograd.html", "note": "API reference for backward, grad, Function, gradcheck and anomaly detection."}, {"title": "Paszke et al.: PyTorch, an imperative style deep learning library (2019)", "url": "https://arxiv.org/abs/1912.01703", "note": "The design rationale for define-by-run autograd and the caching allocator."}]
practice: ["basic-autograd", "gradient-accumulation-equivalence", "softmax", "accumulate-shared-path-gradients"]
tags: ["autograd", "gradient accumulation"]
tools: ["PyTorch"]
---

## A graph built by running your code

PyTorch uses define-by-run reverse-mode automatic differentiation. You never declare a graph. When an operation runs on at least one tensor that requires gradients, PyTorch computes the result and also attaches a `grad_fn` node that knows the operation's backward formula and holds references to whatever it needs (saved inputs or outputs). The Deep Learning course builds a scalar version of this engine from scratch; here we look at how PyTorch's version behaves.

```python
w = torch.tensor([2.0, -1.0], requires_grad=True)   # leaf
x = torch.tensor([3.0, 4.0])                        # no grad needed
y = (w * x).sum()          # y.grad_fn = <SumBackward0>
z = y ** 2                 # z.grad_fn = <PowBackward0>
z.backward()
w.grad                     # tensor([12., 16.])  = 2*y*x with y = 2
```

**Leaves** are tensors you created directly (parameters, inputs). Only leaves with `requires_grad=True` get `.grad` populated. Intermediate tensors have `grad_fn` but their `.grad` stays `None` unless you call `.retain_grad()` on them, which is useful for debugging.

~~~dg-flow Forward builds the graph; backward walks it in reverse and accumulates into leaves.
w (leaf) -(mul x)-> MulBackward -(sum)-> SumBackward -(pow 2)-> *z -(backward)-> w.grad += dz/dw
~~~

## What backward actually computes

For a scalar loss $L$, `L.backward()` computes $\partial L / \partial \theta$ for every leaf $\theta$ by the chain rule, one node at a time. Each node does not form its full Jacobian; it computes a **vector-Jacobian product**: given the incoming gradient $v = \partial L / \partial \text{out}$, it returns $v^\top J$ for each input. That is why calling `backward()` on a non-scalar tensor raises `grad can be implicitly created only for scalar outputs`: PyTorch needs the vector $v$, which you can pass as `out.backward(gradient=v)`.

The backward pass also frees each node's saved tensors as it goes, so peak memory drops after `backward()`. If you need to backpropagate through the same graph twice (rare: multiple losses sharing a trunk are usually summed first), pass `retain_graph=True`. For gradients of gradients (gradient penalties, meta-learning, Hessian-vector products) use `torch.autograd.grad(..., create_graph=True)`, which records the backward pass itself as a graph.

## Gradients accumulate

`backward()` adds into `.grad`; it never overwrites. That has two faces.

- **The bug**: forgetting `optimizer.zero_grad()` means step $k$ uses the sum of all gradients so far. Loss usually falls for a few steps and then diverges.
- **The feature**: if one batch of 256 does not fit in memory, run four micro-batches of 64, call `backward()` on each with the loss divided by 4, and step once. The accumulated gradient equals the large-batch gradient (exactly, if every micro-batch has the same number of loss terms; with padded sequences, divide by the total token count instead).

```python
opt.zero_grad(set_to_none=True)    # default since 2.0: frees .grad instead of filling zeros
for xb, yb in micro_batches:       # 4 chunks of one logical batch
    loss = loss_fn(model(xb), yb) / len(micro_batches)
    loss.backward()                # grads add up across chunks
opt.step()
```

A shared parameter used twice in one forward pass (tied embeddings, a recurrent cell) also accumulates: autograd sums the contributions from both paths, which is exactly the multivariable chain rule.

## Switching recording off

| tool | effect | use it for
| --- | --- | --- |
| `torch.no_grad()` | ops inside are not recorded; outputs have `requires_grad=False` | validation, optimiser updates, manual weight edits
| `torch.inference_mode()` | like `no_grad` but also skips version counting; faster, and its tensors can never enter autograd later | pure inference
| `t.detach()` | new tensor, same storage, no history | stopping gradients (targets, EMA teachers, logging)
| `p.requires_grad_(False)` | parameter becomes a constant leaf | freezing layers for fine-tuning

`detach()` shares storage, so an in-place edit to the detached tensor also edits the original and can trip the version-counter check from the previous lesson. `model.eval()` does **not** stop gradient recording; it only changes layer behaviour (dropout, batch norm). Validation code needs both `model.eval()` and `no_grad()`.

## Custom backward rules

When an op has a better backward than autograd would derive (numerical stability, memory), subclass `torch.autograd.Function` and write `forward` and `backward` static methods, saving only what you need with `ctx.save_for_backward`. Check it with `torch.autograd.gradcheck` in float64 against finite differences.

<div class="ex">
<div class="lab">Worked example</div>

**why the logged loss ate your memory**

A loop appends `loss` to a Python list every step to plot it later. After a few thousand steps the GPU runs out of memory.

Each `loss` still references its `grad_fn`, so the list keeps every step's graph metadata and the tensors it saved alive. Append `loss.item()` (a Python float, at the cost of a sync) or `loss.detach()` instead.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

autograd records what you run, sums gradients into leaves, and frees the graph after one backward. Zero gradients each step, wrap non-training code in `no_grad`, and detach anything you keep around.

</div>
