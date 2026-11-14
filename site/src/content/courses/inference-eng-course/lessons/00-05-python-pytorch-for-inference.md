---
id: "00.05"
course: "inference-eng-course"
module: "00"
slug: "python-pytorch-for-inference"
title: "Python & PyTorch for Inference"
keyIdeas: ["Use torch.inference_mode() to skip autograd bookkeeping.", "Choose dtype on load (torch_dtype=torch.bfloat16) and device placement up front.", "CUDA calls are asynchronous. Call torch.cuda.synchronize() before reading a timer.", "torch.cuda.memory_allocated and max_memory_allocated show what the model and activations really use.", "Python overhead matters at small batch sizes; engines hide it with CUDA graphs and compiled code."]
question: "Name three PyTorch settings or habits that matter for inference but not for training, and say why."
answer: "Use torch.inference_mode() (or no_grad) so autograd does not store activations. Load weights directly in BF16 or FP16 on the GPU instead of FP32 on CPU, halving memory and load time. Synchronize with torch.cuda.synchronize() before taking timings, because CUDA calls are asynchronous and timing without it measures only kernel launch."
sources: [{"title": "PyTorch: inference_mode", "url": "https://pytorch.org/docs/stable/generated/torch.inference_mode.html", "note": "Disables autograd tracking for inference."}, {"title": "PyTorch: CUDA semantics", "url": "https://pytorch.org/docs/stable/notes/cuda.html", "note": "Asynchronous execution, streams and memory management."}]
tags: ["pytorch"]
---

## The PyTorch you actually need

You do not need to be a PyTorch expert to do inference work, but you need a handful of habits that differ from training code. Most slow or wrong inference scripts break one of these.

## Turn off training machinery

```
import torch
model.eval()                      # disable dropout, use inference behavior
with torch.inference_mode():      # no autograd graph, no saved activations
    out = model(input_ids)
```

`inference_mode` is stricter and faster than `no_grad`: tensors created inside cannot be used in autograd later, so PyTorch skips version tracking. Without it, a forward pass keeps activations alive for a backward pass that never comes, and memory use climbs.

## Pick the dtype on purpose

```
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
```

Loading in FP32 by accident doubles memory and halves speed. BF16 is the default for modern GPUs (Ampere and newer). FP16 has more mantissa bits but a smaller range and can overflow in activations of large models.

## Devices and transfers

Tensors live on a device. Moving data between CPU and GPU crosses PCIe at roughly 25 GB/s, about 100× slower than GPU memory. Keep everything on the GPU in the hot loop. A classic mistake is calling `.item()` or `.cpu()` every token to check for a stop condition; each call forces the CPU to wait for the GPU to finish (a synchronization), which removes overlap.

## CUDA is asynchronous

When Python calls a GPU operation, it enqueues a kernel and returns immediately. Timing code that does not synchronize measures only launch time.

```
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    model(x)
torch.cuda.synchronize()
print((time.perf_counter() - t0) * 1000, "ms")
```

<div class="ex">
<div class="lab">Worked example</div>

**a misleading timer**

Without synchronize, a 7B forward pass "takes" 4 ms: the time Python needed to launch about 600 kernels.

With synchronize, the same pass takes 48 ms. The 44 ms difference is the GPU doing the actual work after Python moved on.

Use CUDA events or the profiler for kernel-level timing; use synchronize plus perf_counter for end-to-end.

</div>

## Memory: allocated versus reserved

PyTorch's caching allocator keeps freed blocks to reuse. `torch.cuda.memory_allocated()` reports bytes held by live tensors. `memory_reserved()` reports what the allocator has taken from the driver. nvidia-smi shows reserved plus the CUDA context (300 to 600 MB). If nvidia-smi says 20 GB but allocated says 15 GB, the gap is cache and context, not a leak.

## Python overhead is real at batch 1

A 32-layer model launches several hundred kernels per forward pass. Each launch costs a few microseconds of CPU time. At batch 1 on a fast GPU, a step might be 6 ms of GPU work and 3 ms of launch overhead. That is why engines use CUDA graphs and torch.compile, which you will study later.

## A minimal inference checklist

- `model.eval()` and `torch.inference_mode()`.
- Explicit dtype (BF16) and device.
- No per-token `.item()` or `.cpu()` in the loop.
- Synchronize before reading timers.
- Watch `memory_allocated` and `max_memory_allocated` after warm-up.

## Deep dive: when Python becomes the bottleneck

Each CUDA kernel launch costs the CPU about 4 to 10 µs. A decode step of a 32-layer model with no fusion launches roughly 15 kernels per layer, or about 500 per step:

⟦M131⟧

If the GPU work for that step is only 5 ms, the CPU barely keeps ahead. If Python adds its own overhead per layer (module calls, shape checks), the GPU starts waiting on the CPU, and you see gaps between kernels in a profile.

~~~dg-raw
%%timeline CPU launching kernels ahead of a GPU (top) versus a GPU starved by a slow CPU (bottom).
unit = ms
CPU (fast) | 0-3 launch step 1 c4 ; 3-6 launch step 2 c4
GPU (fed) | 0.2-5.2 step 1 c1 ; 5.2-10.2 step 2 c1
CPU (slow) | 0-7 launch step 1 c4 ; 7-14 launch step 2 c4
GPU (starved) | 0.3-7.2 step 1 (gaps) flame ; 7.2-14.2 step 2 (gaps) flame
%%
~~~

Because launches are asynchronous, the CPU normally runs ahead of the GPU and the launch cost is hidden. It shows up only when the CPU cannot stay ahead. CUDA graphs fix this by replaying a whole step's launches with one call.

## Deep dive: measuring memory bandwidth yourself

A copy of an array of ⟦m133⟧ bytes reads ⟦m134⟧ and writes ⟦m135⟧, so

⟦M132⟧

```
x = torch.empty(2**30, dtype=torch.uint8, device="cuda")   # 1 GiB
y = torch.empty_like(x)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): y.copy_(x)
torch.cuda.synchronize(); t = (time.perf_counter() - t0) / 20
print(2 * x.numel() / t / 1e12, "TB/s")
```

An H100 SXM typically reports 2.8 to 3.0 TB/s here, about 85 to 90% of the 3.35 TB/s on the spec sheet. Use the measured value, not the spec, in your estimates.

<div class="take">
<div class="lab">Key takeaway</div>

inference-mode PyTorch is about avoiding hidden work: autograd, wrong dtypes, CPU-GPU syncs and unsynchronized timers.

</div>
