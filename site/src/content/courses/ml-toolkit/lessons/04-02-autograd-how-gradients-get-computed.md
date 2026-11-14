---
id: "04.02"
course: "ml-toolkit"
module: "04"
slug: "autograd-how-gradients-get-computed"
title: "Autograd: how gradients get computed"
keyIdeas: ["Autograd records a graph *while* your forward code runs: every operation on a tensor with `requires_grad=True` produces an output whose `grad_fn` points back at its inputs. The graph is rebuilt every forward pass, so ordinary Python `if` and `for` just work.", "`loss.backward()` walks that graph in reverse, applying each operation's local derivative rule (a vector-Jacobian product), and leaves the gradient in `.grad` of each leaf. For $z = (w \\cdot x)^2$ with $w = (2, -1)$, $x = (3, 4)$ it gives $(12, 16)$: the slope of the loss along each weight.", "Backward **adds** into `.grad` and never overwrites. Forgetting `zero_grad()` therefore piles up old gradients, while the same rule makes gradient accumulation work: divide each micro-batch loss by the number of chunks and step once.", "After `backward()` the graph's saved tensors are freed. A second backward through the same graph needs `retain_graph=True`; gradients of gradients need `torch.autograd.grad(..., create_graph=True)`.", "`torch.no_grad()` and `torch.inference_mode()` stop recording (for evaluation and optimiser updates), `.detach()` cuts one tensor out of the graph while sharing its storage, and `requires_grad_(False)` freezes parameters. `model.eval()` does not stop recording.", "Keeping loss tensors around (`total += loss`) keeps their whole graphs alive and leaks memory; log `loss.item()` or `loss.detach()` instead."]
question: "You train with gradient accumulation over 8 micro-batches but write `loss = loss_fn(out, y); loss.backward()` without dividing by 8, and you also call `opt.zero_grad()` inside the micro-batch loop instead of before it. Describe precisely what gradient each optimiser step uses, and what each bug alone would do to training. Your validation loop also does `val_total += loss` without `torch.no_grad()`; what happens there?"
answer: "With `zero_grad()` inside the loop, each micro-batch's gradients are wiped before the next one is added, so when `opt.step()` runs, `.grad` holds only the last micro-batch's gradient (and because that loss was not divided by 8, it is the plain mean-loss gradient of that one chunk). Training then behaves like batch size B/8: 7/8 of the data's gradient signal is thrown away and the step has the noise of a small batch, even though you pay the compute for all 8 chunks. Without that bug, forgetting the division makes the step use the sum rather than the mean of the 8 micro-batch gradients: the direction is right but the magnitude is 8 times too large. For plain SGD that is the same as multiplying the learning rate by 8, which can make training unstable; for Adam the scale largely cancels because it divides by the root of the second-moment estimate, though epsilon and weight-decay interactions shift slightly, and any gradient-clipping threshold is now wrong by a factor of 8. The fixes: call `opt.zero_grad()` once before the micro-batch loop, and divide each micro-batch loss by 8 (or by the total number of loss terms, such as tokens, in the logical batch). In the validation loop, the forward pass is recorded because no `no_grad()` is active, and `val_total += loss` makes `val_total` a tensor that depends on every batch's graph, so all their saved activations stay alive until the end of validation. Memory grows linearly with the number of batches and can run out on a large validation set. Wrap the loop in `torch.no_grad()` (or `inference_mode()`), call `model.eval()` as well, and accumulate `loss.item()`."
sources: [{"title": "PyTorch docs: Autograd mechanics", "url": "https://pytorch.org/docs/stable/notes/autograd.html", "note": "How the graph is recorded, saved tensors, in-place correctness checks and grad modes."}, {"title": "PyTorch tutorial: A gentle introduction to torch.autograd", "url": "https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html", "note": "Vector-Jacobian products and the DAG, with small runnable examples."}, {"title": "PyTorch docs: torch.autograd", "url": "https://pytorch.org/docs/stable/autograd.html", "note": "API reference for backward, grad, Function, gradcheck and anomaly detection."}, {"title": "Paszke et al.: PyTorch, an imperative style deep learning library (2019)", "url": "https://arxiv.org/abs/1912.01703", "note": "The design rationale for define-by-run autograd and the caching allocator."}]
practice: ["basic-autograd", "gradient-accumulation-equivalence", "softmax", "accumulate-shared-path-gradients"]
tags: ["autograd", "gradient accumulation"]
tools: ["PyTorch"]
---

## What you will learn

- What PyTorch's **autograd** does: it records the operations you run and then replays them backwards to get gradients, with no graph for you to declare.
- How to read the recorded graph (`grad_fn`, leaves, `.grad`), follow a backward pass by hand, and write your own backward rule.
- Why `backward()` **adds** into `.grad`, how that causes the classic "forgot `zero_grad`" bug, and how the same rule gives you gradient accumulation for free.
- When the graph is freed, and what `retain_graph=True`, `create_graph=True` and `torch.autograd.grad` are for.
- How to switch recording off (`no_grad`, `inference_mode`, `detach`, `requires_grad_(False)`), and the memory leaks that happen when you forget.

*Before you start:* you need tensors, dtypes and in-place operations from lesson 04.01, and the idea that training means "nudge each weight against the slope of the loss" (gradient descent). The chain rule helps; the Deep Learning course builds a tiny scalar autograd engine from scratch if you want to see one from the inside.

## From tensors to gradients

Lesson 04.01 treated tensors as fast arrays with a dtype and a device. One more flag, `requires_grad=True`, asks PyTorch to remember every operation that touches a tensor, so that later you can ask "how does the loss change if I nudge this tensor a little?". Modules (04.03) and optimisers (04.04) are built on the answer.

## The idea in one picture

Take two weights and two inputs:

```python
import torch
w = torch.tensor([2.0, -1.0], requires_grad=True)   # weights we want to learn
x = torch.tensor([3.0, 4.0])                        # data: no gradient needed
z = ((w * x).sum()) ** 2                            # a tiny "loss"
z.backward()
print(z.item(), w.grad)                             # 4.0  tensor([12., 16.])
```

The forward pass computes $w \cdot x = 2 \times 3 + (-1) \times 4 = 2$, then squares it to get $z = 4$. The call `z.backward()` fills `w.grad` with two numbers, 12 and 16. What do they mean?

~~~dg-fig 04.02-slope.svg
Hold the second weight at -1 and let only the first one vary: the loss traces a parabola. The gradient autograd stored, 12, is the slope of the tangent line at the current weight w1 = 2. Stepping against that slope lowers the loss.
~~~

`w.grad[0] = 12` says: if you increase the first weight by a tiny amount $\epsilon$, the loss rises by about $12\epsilon$. You can check it numerically. Moving $w_1$ from 2 to 2.001 changes $w \cdot x$ from 2 to 2.003 and $z$ from 4 to 4.012009, a rise of 0.012, which is $12 \times 0.001$ to within rounding. Likewise `w.grad[1] = 16` is the slope along the second weight. Together they form the **gradient**: the direction in which the loss rises fastest. Gradient descent steps the opposite way.

Two words carry the rest of the lesson:

- **Autograd** is PyTorch's automatic differentiation engine. It neither does symbolic algebra nor approximates with finite differences; it applies the exact derivative rule of each operation, in order.
- **Define-by-run** means the graph is recorded *while your Python code runs*, fresh on every forward pass, so `if` and `for` in your model just work: whatever actually ran is what gets differentiated.

## Step 1: the forward pass records a graph

Whenever an operation runs and at least one of its inputs requires a gradient, PyTorch does two things: it computes the result as usual, and it attaches to that result a small object called a **grad_fn** node. The node knows which operation produced the tensor, holds the derivative rule for it, and keeps references to whatever that rule will need later (for a multiplication, the other factor). Let us split the example into named steps and look:

```python
p = w * x          # elementwise product
y = p.sum()        # scalar
z = y ** 2
print(w.is_leaf, p.is_leaf)   # True False
print(p.grad_fn, y.grad_fn, z.grad_fn)
# <MulBackward0 ...> <SumBackward0 ...> <PowBackward0 ...>
print(z.grad_fn.next_functions)   # ((<SumBackward0 ...>, 0),)  -> each node points back at its inputs
```

A **leaf** is a tensor you created directly (parameters, inputs) rather than computed from tracked tensors. Leaves have no `grad_fn`. Every computed tensor has one, and following the `next_functions` links from `z` walks you back through `PowBackward0`, `SumBackward0`, `MulBackward0` and finally to an `AccumulateGrad` node that belongs to the leaf `w`. The tensor `x` has `requires_grad=False`, so its slot in that list is simply `None`: autograd will not waste time computing a gradient nobody asked for.

~~~dg-fig 04.02-graph.svg
The graph recorded for z = (w·x)². Top arrows: the forward pass computes values left to right and stores what each backward rule needs (here x). Bottom arrows: backward starts from dz/dz = 1 and pushes gradients right to left until they land in w.grad.
~~~

## Step 2: backward walks the graph in reverse

`z.backward()` starts at the output with the trivial gradient $\partial z / \partial z = 1$ and visits the nodes in reverse order. Each node receives "how much the final loss changes per unit change of my output" and turns it into "how much the loss changes per unit change of each of my inputs". This is the **chain rule**, applied one link at a time:

$$\frac{\partial z}{\partial w} = \frac{\partial z}{\partial y}\cdot\frac{\partial y}{\partial p}\cdot\frac{\partial p}{\partial w}.$$

Here $z$ is the loss, $y = \sum_i p_i$ is the sum, $p = w \odot x$ is the elementwise product, and $w$ is the leaf we care about. Each factor is the local derivative rule of one node.

<div class="ex">
<div class="lab">Worked example</div>

**run the backward pass by hand**

Forward values: $p = w \odot x = (2 \times 3,\; -1 \times 4) = (6, -4)$. Then $y = 6 + (-4) = 2$, and $z = y^2 = 4$.

Start at the top: $\partial z / \partial z = 1$.

PowBackward0 (rule for $z = y^2$ is $2y$): $\partial z / \partial y = 1 \times 2y = 2 \times 2 = 4$.

SumBackward0 (each input of a sum contributes with slope 1, so the incoming gradient is copied to every element): $\partial z / \partial p = (4 \times 1,\; 4 \times 1) = (4, 4)$.

MulBackward0 (rule for $p_i = w_i x_i$ with respect to $w_i$ is $x_i$, which is why $x$ was saved): $\partial z / \partial w = (4 \times 3,\; 4 \times 4) = (12, 16)$.

AccumulateGrad adds $(12, 16)$ into `w.grad`, which was empty, so `w.grad` is now `tensor([12., 16.])`. This matches the printed result and the tangent slope in the first figure.

</div>

No node ever builds a full matrix of derivatives. Each one takes the incoming gradient vector $v$ and returns $v^\top J$, where $J$ is that node's **Jacobian** (the table of partial derivatives of its outputs with respect to its inputs). This product is called a **vector-Jacobian product (VJP)**. For a layer with a million inputs and outputs the full Jacobian would have $10^{12}$ entries, but the VJP costs about as much as the forward operation. So one backward pass costs a small constant multiple of one forward pass, however many weights there are. This is **reverse-mode** differentiation.

The VJP view also explains a common error. If the output is not a scalar, there is no natural starting vector:

```python
w.grad = None                 # start clean (see step 4 for why)
out = w * x                   # shape (2,), not a scalar
out.backward()                # RuntimeError: grad can be implicitly created only for scalar outputs
out.backward(gradient=torch.tensor([1.0, 1.0]))   # you supply v explicitly
print(w.grad)                 # tensor([3., 4.])  (= v * x)
```

A vector of ones is the same as calling backward on `out.sum()`. In training you reduce to a scalar loss first, so this is rare.

## Step 3: only leaves keep their gradient

After backward, `w.grad` holds a tensor, but `p.grad` and `y.grad` are `None`, and PyTorch warns if you read them. Storing a gradient for every intermediate tensor would double memory for nothing; optimisers only need the parameters' gradients. When you are debugging and want to see the gradient arriving at an intermediate value, ask for it explicitly before calling backward:

```python
w.grad = None
p = w * x; p.retain_grad()        # keep this one
z = (p.sum()) ** 2; z.backward()
print(p.grad)                     # tensor([4., 4.])  matches the hand computation
```

A hook, `p.register_hook(lambda g: print(g))`, shows the same gradient as it flows past without storing anything.

## Step 4: gradients accumulate

Here is the rule that surprises everyone once: **backward adds into `.grad`, it never overwrites**. Run the forward and backward again without clearing, and the gradient doubles:

```python
z = ((w * x).sum()) ** 2       # new forward pass, new graph
z.backward()
print(w.grad)                  # tensor([24., 32.])  = 12+12, 16+16
w.grad = None                  # clear it (what zero_grad(set_to_none=True) does)
```

The chain rule demands it: when one tensor is used along several paths, its total derivative is the sum over the paths, and autograd computes that sum by adding into one buffer:

```python
a = torch.tensor(3.0, requires_grad=True)
f = a * a + 2 * a      # a is used three times
f.backward()
print(a.grad)          # tensor(8.)  = 2a + 2 at a = 3
```

The same mechanism handles shared parameters inside a model, such as an embedding matrix reused as the output layer, or a recurrent cell applied at every time step.

The rule has a bad face and a good face.

**The bug.** Forgetting `optimizer.zero_grad()` means step $k$ uses the sum of all gradients from steps 1 to $k$. Old gradients never decay, so the weights keep moving in directions that were right many steps ago, and the loss swings or diverges.

~~~dg-fig 04.02-zerograd.svg
Measured on a three-weight linear regression with plain SGD (learning rate 0.05). Clearing gradients each step drives the loss down to the noise floor of about 0.01; forgetting to clear makes the weights overshoot again and again, and after 60 steps the loss is still above 1.
~~~

**The feature.** Suppose a batch of 256 examples does not fit in memory. Split it into 4 **micro-batches** of 64, call `backward()` on each, and step the optimiser once. Because each backward adds into `.grad`, the result after four calls is the sum of four micro-batch gradients. Divide each micro-batch loss by 4 and that sum equals the gradient of the mean loss over all 256 examples. This is **gradient accumulation**:

```python
opt.zero_grad(set_to_none=True)     # once, before the chunks (set_to_none is the default since 2.0)
for xb, yb in micro_batches:        # 4 chunks of one logical batch
    loss = loss_fn(model(xb), yb) / len(micro_batches)
    loss.backward()                 # adds into .grad
opt.step()
```

~~~dg-fig 04.02-microbatch.svg
Gradients of the three weights of a Linear(3, 1) layer, measured four ways. Four micro-batches with each loss divided by 4 match one batch of 256 to within 1e-7. Without the division every gradient is exactly four times too big; calling zero_grad inside the loop leaves only the last chunk's gradient, which points somewhere else.
~~~

The equality is exact (up to float rounding) when every micro-batch contributes the same number of loss terms. With padded sequences, where chunks contain different numbers of real tokens, divide by the total token count of the whole logical batch instead.

## Step 5: the graph is freed after backward

To save memory, backward releases each node's saved tensors as soon as it has used them. The graph metadata still exists, but its contents are gone, so a second backward fails:

```python
z = ((w * x).sum()) ** 2
z.backward()
z.backward()   # RuntimeError: Trying to backward through the graph a second time ...
```

Usually the fix is to rerun the forward pass. If you truly need two backward passes through one graph, pass `retain_graph=True` to the first. Two losses sharing a trunk do not need it: add them and call backward once.

Sometimes you need the gradient as a value you can differentiate again: gradient penalties in GANs, meta-learning, or Hessian-vector products. Use `torch.autograd.grad`, which returns gradients instead of storing them in `.grad`, together with `create_graph=True`, which records the backward pass itself as a new graph:

```python
u = torch.tensor(2.0, requires_grad=True)
g = u ** 3
(dg,) = torch.autograd.grad(g, u, create_graph=True)
print(dg)      # tensor(12., grad_fn=<MulBackward0>)   3u^2, still differentiable
(d2,) = torch.autograd.grad(dg, u)
print(d2)      # tensor(12.)                           6u
```

## Gradients in a real loop

Put the pieces together and you have a complete, hand-written training step. Fit $\hat y = w x$ to three points that lie on $y = 2x$:

```python
w = torch.tensor(0.0, requires_grad=True)
xs, ys = torch.tensor([1., 2., 3.]), torch.tensor([2., 4., 6.])
for step in range(3):
    loss = ((w * xs - ys) ** 2).mean()   # forward: records a graph
    loss.backward()                      # backward: fills w.grad
    with torch.no_grad():                # the update itself must not be recorded
        w -= 0.1 * w.grad
    w.grad = None                        # clear for the next step
    print(step, round(loss.item(), 4), round(w.item(), 4))
# 0 18.6667 1.8667
# 1 0.083 1.9911
# 2 0.0004 1.9994
```

Check step 0 by hand. With $w = 0$ the residuals are $(-2, -4, -6)$, so the loss is $(4 + 16 + 36)/3 = 18.67$. The gradient of the mean squared error is the mean of $2(wx_i - y_i)x_i = (-4, -16, -36)$, which is $-18.67$. One step with learning rate 0.1 moves $w$ to $0 - 0.1 \times (-18.67) = 1.867$, the printed value. An optimiser (lesson 04.04) does exactly these two lines, update and clear, for every parameter.

## Switching recording off

Recording costs time and memory, and some code must not be recorded at all:

| Tool | Effect | Use it for
| --- | --- | --- |
| `torch.no_grad()` | ops inside are not recorded; outputs have `requires_grad=False` | validation loops, optimiser updates, manual weight edits
| `torch.inference_mode()` | like `no_grad`, and also skips version counting; slightly faster, but its tensors can never be used in autograd later | pure inference and serving
| `t.detach()` | new tensor sharing the same storage, with no history | stopping gradients: targets, EMA teachers, logging
| `p.requires_grad_(False)` | parameter becomes a constant leaf; no `.grad` is computed | freezing layers when fine-tuning

Three details catch people out.

- `model.eval()` does **not** stop recording. It only switches layers such as dropout and batch norm to their evaluation behaviour. Validation code needs both `model.eval()` and `torch.no_grad()`.
- `detach()` shares storage with the original. An in-place edit of the detached tensor also edits the original, and if autograd saved that original for backward, the version-counter check from lesson 04.01 fires: "one of the variables needed for gradient computation has been modified by an inplace operation".
- The update `w -= lr * w.grad` must sit inside `no_grad()`. Outside it, PyTorch refuses to modify a leaf that requires grad in place, and if it did not, the update would become part of next step's graph.

The most expensive mistake is keeping graphs alive by accident. A validation loop that runs without `no_grad` and sums the loss tensors, `total += loss`, makes `total` depend on every batch's graph, and every graph holds its saved activations.

~~~dg-fig 04.02-memory.svg
Measured process memory during a validation loop over a 4-layer MLP (width 1024, batch 512) on this CPU. Summing loss tensors keeps every batch's graph and activations alive, about 16 MB more per batch; using loss.item() or running under no_grad keeps memory flat. Exact numbers are machine-dependent; the straight-line growth is not.
~~~

## Custom backward rules

Occasionally you know a better backward than autograd would derive: more numerically stable, or cheaper in memory. Subclass `torch.autograd.Function`, write `forward` and `backward` as static methods, and save only what backward needs with `ctx.save_for_backward`. Here the exponential saves its *output*, because the derivative of $e^t$ is $e^t$ itself:

```python
class MyExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t):
        out = t.exp()
        ctx.save_for_backward(out)
        return out
    @staticmethod
    def backward(ctx, grad_out):       # receives v, returns v^T J for each input
        (out,) = ctx.saved_tensors
        return grad_out * out

t = torch.randn(5, dtype=torch.float64, requires_grad=True)
print(torch.autograd.gradcheck(MyExp.apply, (t,)))   # True
```

`gradcheck` compares your backward against finite differences. Always run it in float64: in float32 the finite-difference estimate itself is too noisy to judge anything.

## Checking that gradients are right

When training misbehaves, look at gradients before blaming the model.

- **Present?** After `loss.backward()`, flag any parameter whose `p.grad is None`: it was not on the path to the loss (frozen, unused, or cut by `.detach()` or `.item()`).
- **Sensible?** Print each parameter's gradient norm. All zeros suggest a dead or detached path; huge values suggest exploding gradients (lesson 04.08); `nan` suggests something like `log(0)`.
- **Where did a nan start?** `torch.autograd.set_detect_anomaly(True)` is slow, but its error names the forward operation whose backward produced it.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss falls, then oscillates or explodes | `zero_grad()` missing, gradients pile up | call `opt.zero_grad()` once per optimiser step
| Accumulated training behaves like a much higher learning rate | micro-batch losses not divided by the number of chunks | divide each loss by the chunk count (or total tokens)
| Accumulated training is noisy, as if the batch were tiny | `zero_grad()` inside the micro-batch loop | move it before the loop
| "grad can be implicitly created only for scalar outputs" | backward on a vector | reduce to a scalar loss, or pass `gradient=`
| "Trying to backward through the graph a second time" | reusing a freed graph | rerun the forward, sum the losses, or `retain_graph=True`
| Memory grows every step or batch | storing loss tensors, or validation without `no_grad` | use `loss.item()` or `loss.detach()`; wrap evaluation in `no_grad`
| A parameter's `.grad` is `None` | graph cut by `.detach()`, `.item()`, `.numpy()` or a non-tensor detour | keep the computation in torch ops on tracked tensors
| "modified by an inplace operation" | in-place edit of a tensor saved for backward | use the out-of-place version, or clone first

## Going deeper

**Why reverse mode.** For a function from $n$ inputs to $m$ outputs, reverse mode gets the gradient of one output with respect to all $n$ inputs in one backward pass, while **forward mode** gets the derivative of all $m$ outputs with respect to one input direction in one pass. Training has $n$ in the millions and $m = 1$, so reverse mode wins by a factor of about $n$. The price is memory: the forward pass's intermediate values must be kept until backward uses them, which is why activations dominate training memory and why **activation checkpointing** (`torch.utils.checkpoint`) discards them and recomputes them during backward. Forward mode exists too (`torch.func.jvp`) and wins when outputs outnumber inputs.

**Functional transforms.** `torch.func` offers JAX-style transforms on plain functions: `grad(f)(w)` returns the gradient without touching `.grad`, and composes with `vmap` to get per-example gradients and with `jacrev` or `hessian` for full derivative matrices. The JAX module is built entirely on this style.

**Where accumulation is not exact.** Gradient accumulation reproduces the large-batch gradient for losses that are averages of per-example terms. It does not reproduce layers that look across the batch: batch norm computes its statistics per micro-batch, so 4 chunks of 64 normalise differently from one batch of 256. Contrastive losses that compare every example with every other in the batch also change meaning. For those, use larger real batches, or layers that do not depend on batch statistics (layer norm, group norm).

## Key terms

| Term | Meaning
| --- | --- |
| Autograd | PyTorch's engine that records operations and computes exact gradients by the chain rule
| Define-by-run | The graph is recorded while your Python code runs, rebuilt every forward pass
| Leaf tensor | A tensor created directly (e.g. a parameter); only leaves get `.grad` filled by default
| grad_fn | The node attached to a computed tensor that holds its operation's backward rule
| Vector-Jacobian product | What each node computes in backward: incoming gradient times the node's local derivatives
| Gradient accumulation | Calling backward on several micro-batches before one optimiser step; relies on `.grad` adding up
| `no_grad` / `inference_mode` | Context managers that stop recording, for evaluation and weight updates
| `detach()` | Returns a tensor with the same data but no history, cutting it out of the graph
| `gradcheck` | Compares a backward rule with finite differences, in float64

<div class="take">
<div class="lab">Key takeaway</div>

autograd records what your code actually runs, walks that graph backwards applying one local derivative per operation, and adds the results into each leaf's `.grad`. Clear gradients once per optimiser step, divide micro-batch losses when accumulating, wrap evaluation and updates in `no_grad`, and never keep loss tensors around when a float will do.

</div>
