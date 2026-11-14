---
id: "12.07"
course: "inference-eng-course"
module: "12"
slug: "compute-bound-vs-memory-bound-the-roofline-model"
title: "Compute-Bound vs Memory-Bound & the Roofline Model"
keyIdeas: ["Arithmetic intensity = FLOPs / bytes moved.", "Attainable FLOPs = min(peak FLOPs, intensity x bandwidth).", "Ridge point = peak FLOPs / bandwidth. Below it: memory bound. Above: compute bound.", "Decode GEMMs have intensity close to the batch size; prefill GEMMs close to the token count.", "Use the roofline to know how much speedup is possible before you optimize."]
question: "An H100 has about 990 TFLOPs of dense BF16 and 3.35 TB/s of HBM bandwidth. What arithmetic intensity is the ridge point, and is a batch-8 decode GEMM above or below it?"
answer: "Ridge = 990e12 / 3.35e12, about 295 FLOPs per byte. A BF16 GEMM with M = 8 does 2 x 8 x K x N FLOPs while reading about 2 x K x N bytes of weights, so intensity is about 8 FLOPs per byte, far below the ridge. It is memory bound."
sources: [{"title": "Roofline: An Insightful Visual Performance Model", "url": "https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf", "note": "The original roofline paper."}, {"title": "Horace He: Making Deep Learning Go Brrrr", "url": "https://horace.io/brrr_intro.html", "note": "Compute, bandwidth and overhead."}]
tags: ["roofline"]
---

## The roofline model

The roofline model predicts the best performance a kernel can reach from one number: its **arithmetic intensity**, the FLOPs it performs per byte it moves to or from memory.

attainable FLOP/s = min(peak FLOP/s, intensity × memory bandwidth)

At low intensity, the kernel is limited by bandwidth: each byte supports only a few FLOPs. At high intensity, it is limited by compute. The **ridge point** is the intensity where the two limits meet: peak FLOP/s ÷ bandwidth.

~~~dg-raw
%%widget roof Move along the x-axis and watch the limit change from the memory slope to the flat compute roof. Try different GPUs to see the ridge move.
mode = Decode
n = 6
%%
~~~

## Ridge points for common GPUs

| GPU | Dense BF16 TFLOP/s | HBM TB/s | Ridge (FLOPs per byte) |
| --- | --- | --- | --- |
| A100 80 GB | 312 | 2.0 | about 156 |
| H100 SXM | 990 | 3.35 | about 295 |
| H200 | 990 | 4.8 | about 206 |
| MI300X | 1,300 | 5.3 | about 245 |

Newer GPUs tend to add FLOPs faster than bandwidth, so the ridge moves right, and more kernels become memory bound. The H200's ridge is lower than the H100's because it has the same compute and more bandwidth.

## The intensity of inference operations

For a BF16 matmul of [M, K] by [K, N], where K and N are large and M is small:

- FLOPs: 2 × M × K × N.
- Bytes: about 2 × K × N for the weights, plus small activation reads and writes.
- Intensity: about M FLOPs per byte.

So for linear layers, intensity is roughly the number of tokens in the step.

| Operation | M (tokens) | Regime on an H100 |
| --- | --- | --- |
| Decode, batch 8 | 8 | memory bound |
| Decode, batch 64 | 64 | memory bound |
| Decode, batch 256 | 256 | near the ridge |
| Prefill, 2,048 tokens | 2,048 | compute bound |
| Decode attention (per KV head group) | about the GQA group size, 1 to 8 | memory bound |
| RMSNorm | below 1 | memory bound |

<div class="ex">
<div class="lab">Worked example</div>

**placing a decode matmul on the roofline**

The MLP up-projection of Llama 3 8B at batch 32: [32, 4096] × [4096, 14336].

FLOPs: 2 × 32 × 4,096 × 14,336 ≈ 3.76 GFLOP. Weight bytes: 4,096 × 14,336 × 2 ≈ 117 MB. Intensity ≈ 32.

Roofline limit: 32 × 3.35 TB/s ≈ 107 TFLOP/s, about 11% of peak compute.

Best possible time: 117 MB ÷ 3.35 TB/s ≈ 35 µs. A measured 42 µs means the kernel reaches 83% of its roof. Making it faster needs fewer bytes, such as FP8 weights, not better math.

</div>

## Using the roofline for decisions

Plot each of your top kernels using Nsight Compute's achieved FLOP/s and measured bytes:

- **A kernel far below its roof** has room to improve with a better implementation: better tiling, more parallelism, fewer stalls.
- **A kernel on its roof** cannot get faster without changing the roof itself. For memory-bound kernels, move fewer bytes: lower precision, fusion, smaller KV. For compute-bound kernels, use cheaper math: FP8 or FP4 tensor cores.
- **Raising intensity** moves a kernel right along the slope. Batching does this for decode: more tokens share each weight byte. Speculative decoding does it too, verifying several tokens per weight read.

## Limits of the model

The roofline ignores caches between HBM and the SMs, launch overhead, synchronization and latency effects. Real kernels can sit below the roofline for reasons the model does not show. It remains the best first question to ask of any kernel: which ceiling applies, and how far below it am I?

## Deep dive: the roofline as a formula

Attainable throughput for an operation of intensity ⟦m705⟧ FLOP per byte is

⟦M704⟧

~~~dg-raw
%%chart H100 SXM roofline in BF16 (989 TFLOP/s, 3.35 TB/s). Ridge at 295 FLOP/byte.
x = 1, 4, 16, 64, 128, 295, 1024, 4096
logx = 1
logy = 1
xlab = Arithmetic intensity (FLOP per byte)
ylab = Attainable TFLOP/s
Roof: 3.35, 13.4, 53.6, 214, 429, 989, 989, 989
%%
~~~

| Operation | Intensity | Attainable | Bound |
| --- | --- | --- | --- |
| Decode matmul, batch 1 | 1 | 3.4 TFLOP/s | Memory |
| Decode attention, GQA g = 4 | about 4 | 13 TFLOP/s | Memory |
| Decode matmul, batch 128 | 120 | 400 TFLOP/s | Memory |
| Prefill matmul, 4k tokens | 1,365 | 989 TFLOP/s | Compute |

The ridge moves with precision. In FP8 the compute peak doubles, so the ridge moves to about 590: FP8 helps prefill and big-batch decode, and needs even larger batches before decode becomes compute bound.

<div class="take">
<div class="lab">Key takeaway</div>

arithmetic intensity tells you which ceiling applies. Decode sits below the ridge and wants fewer bytes, prefill sits above and wants cheaper FLOPs, and the distance from a kernel to its roof tells you how much room is left.

</div>
