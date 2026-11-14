---
id: "04.01"
course: "ml-toolkit"
module: "04"
slug: "tensors-devices-and-dtypes"
title: "Tensors, devices and dtypes"
keyIdeas: ["A `torch.Tensor` is the NumPy idea plus two things: a **device** (where the bytes live: `cpu`, `cuda:0`, `mps`) and an optional autograd history. Operations need all their tensor inputs on the same device, and nothing moves between devices unless you call `.to()`.", "Type promotion picks the result dtype by category (bool < integer < float < complex). A Python scalar or 0-dim tensor never upcasts a dimensioned tensor within the same category, so `half_tensor * 2.0` stays float16 but `int_tensor * 0.5` becomes float32.", "Slicing, `view`, `transpose`, `permute`, `expand` and `unsqueeze` return **views** that share storage; `reshape` returns a view when the strides allow it and silently copies otherwise. Advanced (integer or boolean) indexing always copies.", "CUDA kernels launch asynchronously. Calls like `.item()`, `.cpu()`, `print(t)` and `torch.cuda.synchronize()` block until the GPU catches up, which matters both for speed and for honest timing.", "In-place ops (`add_`, `x += 1`, `x[mask] = 0`) save memory but bump a version counter; autograd refuses to backpropagate through a saved tensor that was later modified in place."]
question: "A model runs in bfloat16 on the GPU. Inside `forward` you add a positional table built with `torch.arange(T)[:, None] / 10000 ** (torch.arange(0, d, 2) / d)` and then add it to the bfloat16 activations. Predict the device and dtype of the result, and say what goes wrong and how to fix it."
answer: "`torch.arange` creates int64 tensors on the CPU by default, and the division promotes to the default float dtype, float32. Adding a CPU float32 tensor to CUDA bfloat16 activations fails with a device-mismatch error (PyTorch only lets 0-dim CPU tensors mix with CUDA tensors). If you fix only the device, the addition succeeds, but both operands are dimensioned floats, so the result is promoted to float32: every activation after that point doubles in size and runs slower kernels. Build the table with `device=x.device`, cast it to `x.dtype` (or compute in float32 and cast once), and ideally compute it once in `__init__` and store it with `register_buffer` so that `model.to()` moves and casts it with the rest of the model."
sources: [{"title": "PyTorch docs: Tensor attributes (dtype, device, layout)", "url": "https://pytorch.org/docs/stable/tensor_attributes.html", "note": "The type promotion rules with worked examples, and how devices are named."}, {"title": "PyTorch docs: Tensor views", "url": "https://pytorch.org/docs/stable/tensor_view.html", "note": "The complete list of view operations and the reshape/contiguous caveats."}, {"title": "PyTorch docs: CUDA semantics", "url": "https://pytorch.org/docs/stable/notes/cuda.html", "note": "Asynchronous execution, streams, pinned memory and how the caching allocator behaves."}]
practice: ["tensor-factory", "tensor-operations", "tensor-shape-manipulation", "mutation-trap", "rmsnorm-forward-pass"]
tags: ["dtypes", "devices", "views"]
tools: ["PyTorch"]
---

## Orientation: PyTorch in four layers

This module is about using PyTorch the way production training code does. The AI Research course's *PyTorch Fundamentals* module already walks through creating, indexing, reshaping and stacking tensors, and the NumPy module of this course explains strides and memory layout; here we assume both and focus on what changes once tensors live on accelerators and carry gradients. The rest of the module builds upward: autograd, `nn.Module`, losses and optimisers, data loading, the training loop, speed tricks and debugging.

~~~dg-layers What you touch when you write a training script, from the bottom up.
Tensor | storage + dtype + shape/strides + device
Autograd | records ops on tensors that require grad
nn.Module | holds parameters, buffers and submodules
*Training loop | data, loss, optimiser, checkpoints
~~~

## Devices: where the bytes live

Every tensor has a `device`. Creating it directly on the target device avoids a host-to-device copy:

```python
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(4096, 4096, device=dev)          # allocated on the GPU
w = torch.zeros(10, dtype=torch.float32).to(dev)  # created on CPU, then copied
```

`.to(dev)` returns a new tensor (or `self` if nothing changes); it does not move the original. Mixing devices raises `Expected all tensors to be on the same device`, which is the most common error in a first GPU script. The usual cause is a tensor created inside `forward` without `device=x.device`, or a buffer that was stored as a plain attribute and so was not moved by `model.to(dev)` (lesson 3 fixes this with `register_buffer`).

GPU work is **asynchronous**: Python enqueues a kernel and returns immediately. Code that looks slow at `loss.item()` is usually paying for every kernel queued before it. Two consequences:

- Time GPU code with `torch.cuda.synchronize()` before reading the clock, or with CUDA events.
- Avoid calling `.item()` or `.cpu()` every step in a hot loop; each one is a full pipeline stall.

Host-to-device copies are fastest from **pinned** (page-locked) memory, and with `non_blocking=True` they can overlap with compute. `DataLoader(pin_memory=True)` sets this up for you (lesson 5).

## Dtypes and promotion

The default floating dtype is float32; integer literals give int64. Labels for classification losses must be int64 (`torch.long`), which is why `torch.tensor([0, 2, 1])` works and `torch.tensor([0., 2., 1.])` does not.

| dtype | bytes | exponent / mantissa bits | typical use
| --- | --- | --- | --- |
| float32 | 4 | 8 / 23 | default, optimiser state, master weights
| bfloat16 | 2 | 8 / 7 | LLM training and inference, same range as float32
| float16 | 2 | 5 / 10 | older GPUs, inference; max value 65504
| int64 | 8 | n/a | indices, class labels, token ids
| bool | 1 | n/a | masks

Promotion follows a category rule. When tensors of different dtypes meet, the result takes the highest **category** present (bool, integer, floating, complex). Within a category, a dimensioned tensor beats a 0-dim tensor or Python scalar:

```python
h = torch.ones(3, dtype=torch.float16)
(h * 2.0).dtype                                    # float16: scalar does not upcast
(h + torch.tensor(1.0, dtype=torch.float64)).dtype # float16: 0-dim does not upcast
(h + torch.ones(3)).dtype                          # float32: both dimensioned
(torch.arange(3) * 0.5).dtype                      # float32: int meets float
torch.result_type(h, 2.0)                          # ask without computing
```

Silent promotion is a memory trap: one float32 tensor mixed into a bfloat16 model produces float32 activations from that point on, doubling their memory.

## Views, copies and in-place operations

As in NumPy, a tensor is a header (shape, strides, offset, dtype) over a storage. Basic slicing, `view`, `transpose`, `permute`, `expand`, `unsqueeze` and `narrow` produce new headers over the same storage. Writing through a view changes the original.

- `view` requires strides compatible with the new shape and errors otherwise; after `transpose` you need `.contiguous().view(...)` or `reshape`.
- `reshape` returns a view when it can and a copy when it must, so code that writes through its result has behaviour that depends on the input layout. Do not write through a `reshape` result.
- `expand` creates stride-0 views: no memory, but writing into one is an error because many indices alias one element.
- Integer-array and boolean-mask indexing always copy. `x[mask] = 0` is an in-place write into `x`, but `y = x[mask]; y[:] = 0` changes only the copy.
- `torch.from_numpy(a)` shares memory with the NumPy array; `torch.tensor(a)` copies.

In-place methods end with an underscore (`add_`, `clamp_`, `zero_`), and augmented assignment (`x += 1`) is also in-place. They save an allocation but interact with autograd: every tensor has a version counter, and if a tensor saved for the backward pass is modified afterwards, `backward()` raises `one of the variables needed for gradient computation has been modified by an inplace operation`. In-place ops on a leaf that requires grad are rejected outright. Prefer out-of-place code in models and keep in-place updates for optimiser steps (which run under `no_grad`).

~~~dg-flow Which operations share storage with their input.
x -(slice, view, transpose, expand)-> *view: same storage -(write)-> x changes too
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a mask that leaks**

`labels = tokens; labels[labels == pad_id] = -100` was meant to build loss targets.

Because `labels` is the same tensor as `tokens` (plain assignment copies nothing), the padding ids in `tokens` are overwritten with -100 too, and the next embedding lookup fails with an index error.

The fix is `labels = tokens.clone()` before the in-place write, or the out-of-place `labels = tokens.masked_fill(tokens == pad_id, -100)`.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

know where each tensor lives, what dtype it will be after the next op, and whether it shares storage with something else. Most confusing PyTorch bugs are one of those three.

</div>
