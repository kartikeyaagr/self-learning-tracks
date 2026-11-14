---
id: "11.04"
course: "inference-eng-course"
module: "11"
slug: "quantized-gemm-dequantization-overhead"
title: "Quantized GEMM & Dequantization Overhead"
keyIdeas: ["A quantized GEMM loads low-bit weights, dequantizes in registers, and multiplies.", "Weight-only kernels win when memory bound (small batch), may lose when compute bound.", "W8A8 FP8 or INT8 kernels use low-precision tensor cores, so they also win when compute bound.", "Kernel quality varies; the format's real speed depends on the kernel your engine picks.", "Benchmark at your actual batch sizes, not just batch 1."]
question: "Why can a W4A16 model be faster than BF16 at batch 1 but slower at batch 256?"
answer: "At batch 1 the matmul is memory bound, so reading 4-bit weights cuts time nearly 4 times and dequantization hides under memory latency. At batch 256 the matmul becomes compute bound; the math still runs in BF16 and dequantizing each weight tile adds instructions, so the kernel can run slower than a plain BF16 GEMM."
sources: [{"title": "Marlin: mixed-precision GEMM kernel", "url": "https://github.com/IST-DASLab/marlin", "note": "Fast INT4 x FP16 kernel for batched inference."}, {"title": "Horace He: Making Deep Learning Go Brrrr", "url": "https://horace.io/brrr_intro.html", "note": "Compute vs memory bound."}]
tags: ["kernels", "quantization"]
---

## Inside a weight-only kernel

A W4A16 matmul stores weights in 4 bits but does the multiply in BF16. Each tile of the kernel works like this:

~~~dg-flow One tile of a W4A16 matmul.
Load 4-bit weights -> *Unpack and dequantize in registers -> BF16 tensor core multiply -> Accumulate in FP32
~~~

1. Load packed 4-bit weights from memory: a quarter of the bytes of BF16.
2. Unpack them and multiply by their group scale, in registers.
3. Feed BF16 values to the tensor cores along with the BF16 activations.
4. Accumulate in FP32 and write the output.

Loading is cheap because there are few bytes. Dequantization is extra arithmetic for every weight. Whether that arithmetic costs time depends on the regime.

## Why the gain depends on batch size

- **Memory bound (small batch).** The kernel spends most of its time waiting for weights. The extra dequantization arithmetic fits into that waiting time, so it is free. The kernel runs at the speed of loading a quarter of the bytes: up to about 4 times faster, often 2.5 to 3.5 times in practice.
- **Compute bound (large batch).** The tensor cores are busy. Dequantization now competes with the math for the same SMs, and the matmul is no faster because it still runs in BF16. The kernel can end up the same speed as BF16, or slower.

## The crossover

| Tokens per matmul | BF16 | W4A16 (INT4 weights) | W8A8 FP8 |
| --- | --- | --- | --- |
| 1 to 16 | memory bound | about 2 to 3.5× faster | about 1.8× faster |
| 64 to 128 | mixed | about 1.2 to 2× faster | about 1.5× faster |
| 256 and more | compute bound | about the same or slower | about 1.5 to 2× faster |

These are illustrative numbers. Real values depend on the kernel, the GPU and the matrix shapes. The shape is general: weight-only gains shrink as the batch grows, while FP8 W8A8 gains hold because the math itself is twice as fast.

~~~dg-raw
%%widget roof At low batch, fewer weight bytes moves you up the memory slope. At high batch you are on the compute roof, and only faster math helps.
mode = Decode
n = 4
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why INT4 helped one service and not another**

Service A serves a 70B model to a few users at a time, at batch 4 to 8. INT4 AWQ weights cut TPOT from 28 ms to 11 ms. A clear win.

Service B serves an 8B model at batch 200 for an offline pipeline. INT4 weights changed throughput by −3%: the matmuls were compute bound, and dequantization cost a little. FP8 W8A8 raised throughput by 45%.

Same technique, opposite results, because the batch size put the two services in different regimes.

</div>

## Activation quantization adds its own work

W8A8 kernels must quantize activations at runtime. With per-token dynamic scales, each row of activations is scanned for its maximum, scaled and rounded before the matmul. Good engines fuse this into the previous operation, such as the RMSNorm, so it costs almost nothing. If it runs as a separate kernel, it adds a memory pass and a launch per layer.

## Choosing

- **Low concurrency, memory-limited GPUs** (a 70B model on two GPUs, or any model on a consumer GPU): INT4 weight-only is excellent.
- **High-throughput serving on H100-class GPUs**: FP8 W8A8 is usually the better default, because it helps at every batch size.
- **Blackwell**: FP4 with block scales offers both: fewer bytes and native fast math.
- **Mixed workloads**: measure at your real batch sizes, not at batch 1.

## Check which kernel actually runs

The startup log usually names the quantization kernel. A missing optimized kernel for your GPU, shape or group size can silently fall back to a slow path. Profile one decode step and confirm the matmul kernels have the expected names and timings.

## Deep dive: the batch-size crossover, predicted

Linear-layer time per step, ignoring KV, for an 8B model on an H100 (2.8 TB/s and 600 TFLOP/s achieved in BF16):

⟦M654⟧

W4A16 still multiplies in BF16, plus a dequantization overhead ⟦m655⟧ (about 20% here). FP8 W8A8 halves both the bytes and the math time.

~~~dg-raw
%%chart Predicted linear-layer time per step against batch size.
x = 1, 16, 64, 128, 256, 512
logx = 1
logy = 1
xlab = Batch size
ylab = ms per step
BF16: 5.7, 5.7, 5.7, 5.7, 6.4, 12.8
INT4 weights (W4A16): 1.5, 1.5, 1.9, 3.8, 7.7, 15.4
FP8 (W8A8): 2.9, 2.9, 2.9, 2.9, 3.2, 6.4
%%
~~~

INT4 weights win by almost 4 times at batch 1 and lose above about batch 200. FP8 W8A8 is never worse than BF16. This matches the two services in the example above.

<div class="take">
<div class="lab">Key takeaway</div>

weight-only quantization wins when the kernel is memory bound and fades when it is compute bound. Activation quantization keeps winning at large batch, but only with tensor core support. Benchmark at your real batch sizes and check which kernel runs.

</div>
