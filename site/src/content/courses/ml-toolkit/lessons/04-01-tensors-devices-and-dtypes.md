---
id: "04.01"
course: "ml-toolkit"
module: "04"
slug: "tensors-devices-and-dtypes"
title: "Tensors, devices and dtypes"
keyIdeas: ["A `torch.Tensor` is the NumPy idea (a header of dtype, shape and strides over a flat **storage**) plus a **device** (`cpu`, `cuda:0`, `mps`) and an optional autograd history. Every input of an operation must be on the same device, and nothing moves unless you call `.to()`; on a tensor, `.to()` returns a new tensor rather than changing the old one.", "GPU kernels launch asynchronously. Calls such as `.item()`, `.cpu()` and `print(t)` are synchronisation points that wait for all queued work, so avoid them in hot loops and call `torch.cuda.synchronize()` before reading a clock.", "bfloat16 keeps float32's range with only 7 mantissa bits, so in bfloat16 `1.001` is stored as `1.0` and `256 + 1 = 256`; float16 is more precise but overflows above 65504. Keep sums, counters and optimiser state in float32.", "Type promotion takes the highest category present (bool < integer < float < complex); within it, only dimensioned tensors decide the width. So `half_tensor * 2.0` stays float16, `int_tensor * 0.5` becomes float32, and one float32 tensor mixed into a bfloat16 model promotes everything after it.", "Slicing, `view`, `transpose`, `permute`, `expand` and `detach` return **views** that share storage; `reshape` is a view only when strides allow; integer and boolean indexing always copy. In-place writes (`add_`, `x += 1`, `x[mask] = 0`) bump a version counter, and autograd refuses to backpropagate through a saved tensor that was later modified."]
question: "A model runs in bfloat16 on the GPU. Inside `forward` you add a positional table built with `torch.arange(T)[:, None] / 10000 ** (torch.arange(0, d, 2) / d)` and then add it to the bfloat16 activations. Predict the device and dtype of the result, and say what goes wrong and how to fix it."
answer: "`torch.arange` creates int64 tensors on the CPU by default, and true division promotes them to the default float dtype, float32; the table is a dimensioned (T, d/2) CPU float32 tensor. Adding it to CUDA bfloat16 activations fails with a device-mismatch error (*Expected all tensors to be on the same device*), because only 0-dim CPU tensors may be mixed with CUDA tensors. If you fix only the device with `.to(x.device)`, the addition succeeds, but both operands are dimensioned floating-point tensors, so promotion picks the wider type and the result is float32: every activation after that point takes twice the memory (a (32, 1024, 4096) activation grows from 256 MiB to 512 MiB) and runs slower kernels, with no error to tell you. There is also a precision point: positions such as 257 cannot be represented exactly in bfloat16 (it rounds to 256), so computing the angles in bfloat16 would be wrong; compute the table in float32 and cast the final sin/cos values once. The fix: build the table with `device=x.device`, compute it in float32, cast it with `.to(x.dtype)`, and preferably build it once in `__init__` and store it with `register_buffer` so that `model.to(dev)` moves it with the rest of the model. Check by printing `out.dtype` and `out.device` after the addition, or by watching `torch.cuda.max_memory_allocated()` before and after the fix."
sources: [{"title": "PyTorch docs: Tensor attributes (dtype, device, layout)", "url": "https://pytorch.org/docs/stable/tensor_attributes.html", "note": "The type promotion rules with worked examples, and how devices are named."}, {"title": "PyTorch docs: Tensor views", "url": "https://pytorch.org/docs/stable/tensor_view.html", "note": "The complete list of view operations and the reshape/contiguous caveats."}, {"title": "PyTorch docs: CUDA semantics", "url": "https://pytorch.org/docs/stable/notes/cuda.html", "note": "Asynchronous execution, streams, pinned memory and how the caching allocator behaves."}]
practice: ["tensor-factory", "tensor-operations", "tensor-shape-manipulation", "mutation-trap", "rmsnorm-forward-pass"]
tags: ["dtypes", "devices", "views"]
tools: ["PyTorch"]
---

## What you will learn

- What a PyTorch tensor is: NumPy's array idea (storage, dtype, shape, strides) plus a **device** and an optional gradient history.
- How to put tensors on the right device, why mixing devices fails, and why GPU code runs asynchronously.
- How float32, bfloat16, float16 and int64 differ in memory, range and precision, worked out by hand.
- The type promotion rule that decides the dtype of every result, and how one stray tensor can double your memory.
- Which operations share memory (views) and which copy, and how in-place writes interact with autograd.

*Before you start:* you should know NumPy dtypes and strides (lesson 01.01) and views versus copies (01.02). Basic tensor creation and indexing (the AI Research course's *PyTorch Fundamentals*) is assumed; here we focus on what changes once tensors live on accelerators and carry gradients.

## Where this module starts

This module is about using PyTorch the way production training code does. Everything rests on the tensor, so we start there and build upward: autograd (04.02), `nn.Module` (04.03), losses and optimisers (04.04), `Dataset` and `DataLoader` (04.05), the training loop (04.06), speed tricks such as mixed precision (04.07) and debugging (04.08).

~~~dg-layers What you touch when you write a training script, from the bottom up.
Tensor | storage + dtype + shape/strides + device
Autograd | records ops on tensors that require grad
nn.Module | holds parameters, buffers and submodules
*Training loop | data, loss, optimiser, checkpoints
~~~

Most confusing PyTorch bugs come from losing track of one of three facts about a tensor: *where* it lives, *what type* its numbers are, and *whether it shares memory* with another tensor. This lesson is about all three.

## The idea in one picture

Make a small tensor and ask it to describe itself:

```python
import torch
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(x.dtype, x.shape, x.stride(), x.device, x.requires_grad)
# torch.float32 torch.Size([2, 3]) (3, 1) cpu False
print(x.element_size(), x.nelement() * x.element_size())
# 4 24
```

`dtype`, `shape` and `stride()` work as in NumPy: six 4-byte numbers in one 24-byte block, read row by row. The new fields are `device` (here `cpu`) and `requires_grad` (whether autograd records what happens to this tensor).

~~~dg-fig 04.01-anatomy.svg
One block of memory (the storage, bottom) read through three different headers. x, its transpose and its column 1 all point at the same bytes; the highlighted cells 1, 5 and 9 are literally the same memory, so a write through any of them shows up in all three.
~~~

The words to know:

- The **storage** is the flat block of bytes holding the numbers.
- A **tensor** is a header over a storage: dtype, shape, strides, offset and device.
- A **view** is a second header over the same storage; making one copies no data.
- The **device** is the memory the storage sits in: `cpu`, an NVIDIA GPU (`cuda:0`, ...), or Apple's GPU (`mps`).

## Step 1: create tensors with the dtype and device you mean

Every factory function (`torch.zeros`, `torch.randn`, `torch.tensor`, ...) takes `dtype=` and `device=`. Leave them out and you get defaults:

```python
print(torch.zeros(2, 3).dtype)        # torch.float32  (the default float type)
print(torch.arange(5).dtype)          # torch.int64    (integers default to 64-bit)
print(torch.tensor([0, 2, 1]).dtype)  # torch.int64
print(torch.tensor(3.5).shape)        # torch.Size([])  a 0-dim tensor (a scalar)

import numpy as np
a = np.random.rand(3)                 # NumPy's default float is float64
print(torch.from_numpy(a).dtype)      # torch.float64  shares memory with a
print(torch.tensor(a, dtype=torch.float32).dtype)  # torch.float32, a copy
```

Two defaults bite often. Data loaded through NumPy arrives as float64 and runs at double cost until you cast it. And class labels and token ids must be integers (`torch.long`): an embedding lookup with float indices fails (*... scalar types: Long, Int*), and `CrossEntropyLoss` given labels `[0., 2., 1.]` says it expected a Long target.

When you create a tensor inside a model, copy the dtype and device from a tensor you already have instead of hard-coding them:

```python
z = torch.ones(2, 3, dtype=torch.bfloat16)
print(torch.zeros_like(z).dtype)                         # torch.bfloat16
print(z.new_zeros(5).dtype)                              # torch.bfloat16 (same device too)
noise = torch.randn(2, 3, dtype=z.dtype, device=z.device)
```

The `*_like` functions and `x.new_*` methods inherit both, so the code works unchanged on CPU, GPU and in half precision.

## Step 2: devices and moving data

An operation runs on the device where its inputs live, and PyTorch never moves data behind your back. Every input must be on the same device, or you get the classic first-GPU-script error *Expected all tensors to be on the same device*. (Exception: a 0-dim CPU tensor may be combined with a GPU tensor.)

The standard pattern picks a device once and creates or moves everything onto it:

```python
dev = "cuda" if torch.cuda.is_available() else "cpu"   # or "mps" on Apple silicon
x = torch.randn(4096, 4096, device=dev)   # allocated directly on the target device
w = torch.zeros(10).to(dev)               # created on CPU, then copied over
y = torch.tensor([1.0, 2.0])
print(y.to("cpu") is y)                   # True: already there, so no copy
print(y.to(torch.float64) is y)           # False: a new tensor
```

Two facts about `.to()`:

- On a **tensor**, it returns a new tensor (or the same one if nothing changes). `w.to(dev)` on its own line does nothing useful; write `w = w.to(dev)`.
- On a **module**, `model.to(dev)` moves every registered parameter and buffer in place. A tensor stored as a plain attribute is not registered and stays behind, a classic source of device errors (04.03 fixes it with `register_buffer`).

~~~dg-flow How a batch normally travels from disk to the GPU.
Data on disk -(DataLoader workers)-> CPU tensor -(pin_memory=True)-> Pinned CPU memory -(.to(dev, non_blocking=True))-> *GPU tensor
~~~

CPU-to-GPU copies are fastest from **pinned** (page-locked) memory, which the operating system will not move, so the GPU can read it directly; with `non_blocking=True` the copy overlaps with queued GPU work. `DataLoader(pin_memory=True)` sets this up (04.05).

## Step 3: GPU work is asynchronous

When Python calls a GPU operation, PyTorch puts a job (a *kernel*) in a queue and returns immediately. Python races ahead queueing more work while the GPU catches up. That keeps the GPU busy, with two surprising consequences.

Calls that need an actual number on the CPU must wait for the queue to drain: `.item()`, `.cpu()`, `.tolist()`, `print(t)`, and Python `if` on a tensor value. Each is a **synchronisation point**. A `loss.item()` every step makes the CPU wait for the whole step, removing the overlap; log every 50 or 100 steps instead.

The second is timing: clock a matmul naively and you measure how long it took to *queue* it. Honest GPU timing synchronises first:

```python
import time
def timed(fn, dev):
    if dev == "cuda": torch.cuda.synchronize()   # wait for earlier work
    t0 = time.perf_counter()
    out = fn()
    if dev == "cuda": torch.cuda.synchronize()   # wait for this work
    return out, time.perf_counter() - t0
```

`torch.utils.benchmark.Timer` does this with less noise. CPU code runs synchronously.

## Step 4: dtypes, memory and precision

A floating-point number is stored as a sign, an **exponent** (which power of two it is near) and a **mantissa** (the digits). The exponent's width sets the *range*; the mantissa's width sets the *precision*, how finely nearby numbers are told apart.

~~~dg-fig 04.01-formats.svg
Bit layouts. bfloat16 is float32 with the last 16 mantissa bits cut off: same range, far fewer digits. float16 spends more bits on the mantissa and fewer on the exponent, so it is more precise than bfloat16 but overflows above 65504.
~~~

| dtype | bytes | exponent / mantissa bits | largest value | typical use
| --- | --- | --- | --- | --- |
| float32 | 4 | 8 / 23 | about 3.4e38 | default; optimiser state and master weights
| bfloat16 | 2 | 8 / 7 | about 3.4e38 | training and inference of large models
| float16 | 2 | 5 / 10 | 65504 | older GPUs, inference, mixed precision with loss scaling
| int64 (`long`) | 8 | n/a | about 9.2e18 | indices, class labels, token ids
| bool | 1 | n/a | n/a | masks

Halving the bytes halves memory and roughly doubles throughput, which is why large models train in 16-bit formats. The price is precision. The gap between 1 and the next representable number is called **machine epsilon** (`torch.finfo(dtype).eps`): about $1.2 \times 10^{-7}$ in float32 but $2^{-7} = 0.0078125$ in bfloat16. The gap grows in proportion to the size of the number.

<div class="ex">
<div class="lab">Worked example</div>

**what bfloat16 and float16 actually store**

Numbers between 1 and 2 are spaced $2^{-m}$ apart, where $m$ is the number of mantissa bits. Numbers between $2^e$ and $2^{e+1}$ are spaced $2^{e} \times 2^{-m}$ apart. Every value is rounded to the nearest point on that grid (ties go to the even neighbour).

**1.001 in bfloat16** ($m = 7$): spacing $2^{-7} = 0.0078125$. Distance above 1 in steps: $0.001 / 0.0078125 = 0.128$, which rounds to 0 steps. Stored value: exactly $1.0$. The 0.001 is gone.

**1.001 in float16** ($m = 10$): spacing $2^{-10} = 0.0009765625$. Steps: $0.001 / 0.0009765625 = 1.024$, rounds to 1. Stored value: $1 + 0.0009765625 = 1.0009765625$.

**0.1 in bfloat16**: 0.1 lies between $2^{-4} = 0.0625$ and $2^{-3} = 0.125$, so the spacing is $2^{-4} \times 2^{-7} = 2^{-11} = 0.00048828125$. Steps from zero: $0.1 / 0.00048828125 = 204.8$, rounds to 205. Stored value: $205 \times 0.00048828125 = 0.10009765625$, about 0.1% too large.

**257 in bfloat16**: between $2^8 = 256$ and $2^9 = 512$ the spacing is $256 \times 2^{-7} = 2$, so the grid is 256, 258, 260, ... and 257 is exactly halfway. The tie goes to the even mantissa, 256. So in bfloat16, $256 + 1 = 256$: a counter kept in bfloat16 stops counting at 256.

**Memory**: an activation of shape (32, 1024, 4096) has $32 \times 1024 \times 4096 = 134{,}217{,}728$ elements. At 2 bytes that is $268{,}435{,}456$ bytes = 256 MiB; at 4 bytes (float32) it is 512 MiB.

</div>

You can confirm every number:

```python
for v in [1.001, 0.1, 257.0, 70000.0]:
    print(v, torch.tensor(v, dtype=torch.bfloat16).item(),
             torch.tensor(v, dtype=torch.float16).item())
# 1.001   1.0            1.0009765625
# 0.1     0.10009765625  0.0999755859375
# 257.0   256.0          257.0
# 70000.0 70144.0        inf
```

The same rounding breaks long sums. Once a running total's gap exceeds twice the increment, adding the increment rounds back to the old total.

~~~dg-fig 04.01-precision.svg
Left: the gap between neighbouring numbers grows with magnitude; bfloat16's gaps are 65,536 times float32's. Right: adding 0.01 five thousand times (measured here). bfloat16 stalls at 4 and float16 at 32, where both have a gap of 0.03125; float32 reaches 50.
~~~

This is why mixed-precision training keeps float32 master weights and does reductions and optimiser updates in float32 (04.07). Casts are explicit: `.to(dtype)`, `.float()`, `.bfloat16()`, `.long()`. Float-to-integer casts truncate toward zero (`-2.7` becomes -2), and out-of-range integer casts wrap silently: `torch.tensor([300, -1]).to(torch.uint8)` gives `[44, 255]`.

## Step 5: type promotion decides the result dtype

When an operation mixes dtypes, a two-part rule picks the result's dtype.

1. Rank the *categories*: bool < integer < floating point < complex. The result is in the highest category present.
2. Within that category, only **dimensioned** tensors (with at least one axis) vote on the width. A Python number or a 0-dim tensor can lift the category but never widens the type within it. If no dimensioned tensor is in the winning category, the scalar's category is used at its default width: float32 for floats, int64 for integers.

```python
h = torch.ones(3, dtype=torch.float16)
print((h * 2.0).dtype)                                    # float16: scalar, same category
print((h + torch.tensor(1.0, dtype=torch.float64)).dtype) # float16: 0-dim, same category
print((h + torch.ones(3)).dtype)                          # float32: both dimensioned
print((torch.arange(3) * 0.5).dtype)                      # float32: int meets float
print((torch.arange(3) / 2).dtype)                        # float32: true division
print((torch.arange(5) // 2).dtype)                       # int64: floor division stays integer
print(torch.result_type(h, 2.0))                          # float16: ask without computing
```

~~~dg-fig 04.01-promotion.svg
Result dtypes computed with torch.result_type. Read along a row: Python scalars and 0-dim tensors (left three columns) leave a float tensor's dtype alone, while any dimensioned tensor of a wider or higher type promotes it (bold). Note that bf16 + fp16 gives fp32, because neither can hold the other.
~~~

Number check from the grid: int32 tensor times 2.0 is float32 (float category wins, at the default width); bfloat16 tensor plus a float32 tensor of shape (3,) is float32 (both dimensioned, wider wins).

The rule exists so that `h * 2.0` does not silently turn a half-precision model into float32. The flip side is a memory trap: one dimensioned float32 tensor (a mask, a positional table built with the default dtype) mixed into a bfloat16 model makes every later activation float32, doubling its memory. In-place ops cast back to the target's dtype (`h += torch.ones(3)` stays float16), and raise an error if that would lower the category (`int_tensor += 0.5`).

## Step 6: views, copies and in-place writes

As in NumPy, many operations only make a new header over the same storage. These return **views**: basic slicing (`x[1:]`, `x[:, 0]`), `view`, `transpose`, `permute`, `expand`, `unsqueeze`, `squeeze`, and `detach` (shares memory, drops the gradient history). These always copy: integer-list or tensor indexing (`x[[0, 2]]`), boolean masks (`x[x > 3]`), `clone`, and `contiguous` on a non-contiguous tensor.

```python
t = torch.arange(6.0)
v, s, idx, msk = t.view(2, 3), t[::2], t[[0, 2]], t[t > 3]
print([u.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
       for u in (v, s, idx, msk)])
# [True, True, False, False]   view and slice share; index and mask copy
```

~~~dg-fig 04.01-viewcost.svg
Measured on this machine's CPU, one thread (timings are machine-dependent). A view costs under a microsecond at every size because no data moves; clone and .t().contiguous() copy every byte, so their cost grows with the tensor.
~~~

The rules worth memorising:

- `view` needs compatible strides: `x.t().view(-1)` fails (*view size is not compatible with input tensor's size and stride*). Use `reshape` or `.contiguous().view(...)`.
- `reshape` is a view when strides allow and silently copies otherwise, so never write through its result.
- `expand` makes stride-0 views: all rows point at the same numbers, so it costs no memory, and in-place ops on it fail (*more than one element of the written-to tensor refers to a single memory location*).
- `torch.from_numpy(a)` and `t.numpy()` share memory; `torch.tensor(a)` copies. For a tensor that requires grad use `t.detach().cpu().numpy()`.
- Plain assignment (`b = a`) copies nothing: both names point at one tensor.

**In-place operations** end with an underscore (`add_`, `mul_`, `zero_`); `x += 1` and `x[mask] = 0` are in-place too. They save an allocation but change data other code may still need. Each in-place write increments a **version counter** (shared by views). If a tensor saved for the backward pass has a new version at backward time, `backward()` raises *... modified by an inplace operation ... is at version 1; expected version 0 instead*. In-place ops on a leaf that requires grad (a parameter) are refused outright.

```python
p = torch.ones(3, requires_grad=True)
s = p.sigmoid()        # sigmoid saves its output for backward
s.mul_(2)              # in-place: s._version goes from 0 to 1
s.sum().backward()     # RuntimeError: ... modified by an inplace operation
```

Write models out of place (`s = s * 2`); keep in-place updates for optimiser steps under `torch.no_grad()`.

<div class="ex">
<div class="lab">Worked example</div>

**a mask that leaks**

`labels = tokens; labels[labels == pad_id] = -100` was meant to build loss targets that ignore padding.

Plain assignment copies nothing, so `labels` and `tokens` are one tensor. The write puts -100 into `tokens` too, and the next embedding lookup fails with an index error far from the real bug.

Fix: `labels = tokens.clone()` before the in-place write, or the out-of-place `labels = tokens.masked_fill(tokens == pad_id, -100)`.

</div>

## Checking a tensor: a habit

When something looks wrong, print the three facts before reading more code:

```python
def describe(name, t):
    print(f"{name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
          f"contiguous={t.is_contiguous()} grad={t.requires_grad}")
```

- Device errors: describe every input of the failing line; one will be on `cpu`.
- Memory too high: describe activations after each block; find the first float32 in a bfloat16 model.
- Values changing "by themselves": compare `untyped_storage().data_ptr()` of the suspects; equal pointers mean shared memory.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| *Expected all tensors to be on the same device* | a tensor created inside `forward` without `device=`, or a plain attribute not moved by `model.to()` | create with `device=x.device`; use `register_buffer`
| `w.to(dev)` "did nothing" | tensor `.to()` returns a new tensor | `w = w.to(dev)`
| Training much slower than expected | data left as float64 from NumPy; or `.item()` every step | cast with `.float()`; log every N steps
| Activations use twice the memory | one float32 tensor mixed into a bfloat16 model | build tables and masks with `dtype=x.dtype`
| Loss is `inf` or `nan` in float16 | values above 65504 overflow | use bfloat16, or float16 with loss scaling (04.07)
| A running sum or counter stops growing | accumulation in a 16-bit dtype | accumulate in float32
| Data changes unexpectedly | writing through a view, or `b = a` aliasing | `clone()` before writing
| *modified by an inplace operation* | in-place op on a tensor saved for backward | rewrite out of place

## Going deeper

**Strides count elements.** The element at index $(i_0, \dots, i_{d-1})$ lives at storage position $\text{offset} + \sum_k i_k s_k$, where $s_k$ is the stride of axis $k$. Unlike NumPy, PyTorch counts strides in *elements*, not bytes: a (2, 3) float32 tensor has `stride()` `(3, 1)` where NumPy says `(12, 4)`. A view is a new (shape, stride, offset) triple over the same storage; `x._base` points to the tensor it borrows from.

**Why bfloat16 won for training.** float16 loses precision below about $6 \times 10^{-5}$ (smaller values are subnormal) and overflows above 65504, so small gradients underflow and large activations overflow. Training in float16 needs *loss scaling*: multiply the loss by a large factor before backward, divide the gradients afterwards. bfloat16 has float32's range and usually needs none. Newer hardware adds 8-bit floats (`torch.float8_e4m3fn`, maximum 448; `torch.float8_e5m2`, maximum 57344), used with per-tensor scale factors.

**Hidden precision on GPUs.** `torch.set_float32_matmul_precision("high")` lets float32 matmuls on recent NVIDIA GPUs run as TF32 (10 mantissa bits), much faster; the default `"highest"` does not. Floating-point addition is not associative, so a GPU reduction may differ in the last bits from run to run; `torch.use_deterministic_algorithms(True)` fixes the order at some speed cost.

## Key terms

| Term | Meaning
| --- | --- |
| Storage | The flat block of memory that holds a tensor's numbers
| View | A tensor that reads another tensor's storage through its own shape, strides and offset
| Device | The memory a tensor lives in and where its operations run: `cpu`, `cuda:0`, `mps`
| dtype | The number type of every element, which fixes its size, range and precision
| Machine epsilon | The gap between 1 and the next representable number in a float dtype
| Type promotion | The rule that picks a result's dtype when inputs differ: highest category, dimensioned tensors decide the width
| Synchronisation point | A call such as `.item()` that makes the CPU wait for all queued GPU work
| Pinned memory | Page-locked CPU memory that allows fast and asynchronous copies to the GPU
| Version counter | A per-storage count of in-place writes that autograd checks before backward

<div class="take">
<div class="lab">Key takeaway</div>

for every tensor, know where it lives (device), what its numbers are (dtype, with its range and precision), and whether it shares storage with another tensor. Create tensors with the dtype and device of the data they will meet, accumulate in float32, and write model code out of place.

</div>
