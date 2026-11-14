---
id: "11.01"
course: "inference-eng-course"
module: "11"
slug: "precision-memory-hardware-support"
title: "Precision, Memory & Hardware Support"
keyIdeas: ["Fewer bits per value: less memory, less bandwidth, sometimes faster math.", "Weight-only quantization (W4A16, W8A16) helps memory-bound decode.", "Weight and activation quantization (W8A8, FP8) can also speed compute-bound prefill, if tensor cores support it.", "Hardware support varies: FP8 on Ada/Hopper and newer, FP4 on Blackwell.", "Always measure quality; the cheapest model that fails your eval is not a saving."]
question: "Why does quantizing weights from BF16 to INT4 speed up decode on almost any GPU, but may not speed up prefill?"
answer: "Decode is memory bound, so reading 4 times fewer weight bytes per step directly cuts step time, even if the GPU dequantizes to BF16 before the math. Prefill is compute bound, and if the hardware has no INT4 tensor core path, the math still runs in BF16 plus dequantization overhead, so prefill may be no faster or even slower."
sources: [{"title": "NVIDIA: Hopper architecture tuning guide", "url": "https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html", "note": "Datatype support on H100."}, {"title": "A Survey of Quantization Methods for Efficient Neural Network Inference", "url": "https://arxiv.org/abs/2103.13630", "note": "Background on quantization."}]
tags: ["quantization"]
---

## Why precision matters for serving

Every byte of a weight or a cached key must cross the memory bus, and every multiply runs on hardware built for a particular number format. Fewer bits per number means fewer bytes to store and move, and, if the hardware supports the format, faster math. Quantization is the practice of storing and computing with lower-precision numbers while keeping output quality acceptable.

## The formats

| Format | Bits | Range and precision | Typical use |
| --- | --- | --- | --- |
| FP32 | 32 | wide range, 23 mantissa bits | accumulators, rarely weights |
| BF16 | 16 | FP32's range, 7 mantissa bits | default serving precision |
| FP16 | 16 | narrower range, 10 mantissa bits | older default, overflow risk |
| FP8 E4M3 | 8 | about ±448, 3 mantissa bits | weights and activations on Hopper and newer |
| FP8 E5M2 | 8 | wider range, 2 mantissa bits | gradients in training, sometimes KV |
| INT8 | 8 | 256 evenly spaced levels | weights, or weights and activations with SmoothQuant |
| INT4 | 4 | 16 evenly spaced levels | weights only, with group scales |
| FP4 (MXFP4, NVFP4) | 4 | 16 levels on a log-like scale, block scales | native on Blackwell |

Floating-point formats spend bits on an exponent, so they keep the same relative precision across small and large values. Integer formats space their levels evenly, which suits values with a known, narrow range but handles outliers poorly.

## What gets quantized

Three different things can be stored in lower precision, and they give different benefits:

- **Weights only** (written W4A16 or W8A16). Weights are stored in 4 or 8 bits and converted to BF16 inside the matmul kernel. This saves memory and weight bandwidth, but the math still runs in BF16.
- **Weights and activations** (W8A8, W4A8 or FP4). Both inputs to the matmul are low precision, and the tensor cores run the low-precision math directly. This saves bandwidth and also makes the math faster.
- **KV cache.** Keys and values are stored in FP8 or INT8. This doubles KV capacity and halves attention's memory traffic.

## Hardware support decides speed

Tensor cores accelerate only the formats they implement:

| GPU generation | Native tensor core formats (dense) |
| --- | --- |
| Ampere (A100) | FP16, BF16, TF32, INT8, INT4 |
| Ada (L40S, RTX 4090) | adds FP8 |
| Hopper (H100, H200) | FP8, BF16, FP16, INT8 |
| Blackwell (B200) | adds FP4 and FP6 |

On a GPU without native support, a format can still save memory, but the kernel must convert to a supported format before the math. For example, FP8 weights on an A100 save memory but run BF16 math after conversion.

<div class="ex">
<div class="lab">Worked example</div>

**what FP8 buys on an H100**

An 8B model in BF16: 16 GB of weights, 990 TFLOP/s dense BF16 peak.

In FP8 W8A8: 8 GB of weights, and the FP8 tensor cores peak at about 1,980 TFLOP/s.

Batch-1 decode reads half the bytes, so the ceiling rises from about 210 to about 420 tokens per second.

Prefill can run up to about twice as fast, because the matmuls themselves are faster.

The freed 8 GB holds about 62,000 more tokens of BF16 KV cache.

</div>

## Where the speed comes from, by phase

| Phase | Limited by | Gains most from |
| --- | --- | --- |
| Decode at small batch | weight bandwidth | fewer weight bits |
| Decode at large batch or long context | weight and KV bandwidth, some compute | fewer weight and KV bits |
| Prefill | compute | low-precision tensor core math |

This table explains a common surprise. INT4 weight-only quantization can make batch-1 decode about 3 times faster and prefill no faster at all, because prefill is compute bound and the math still runs in BF16.

## Deep dive: what the bits mean

A floating-point number with ⟦m629⟧ exponent bits and ⟦m630⟧ mantissa bits stores

⟦M628⟧

The exponent sets the range; the mantissa sets the precision. Relative rounding error is at most ⟦m631⟧.

~~~dg-raw
%%grid Bit layouts. Each cell is one bit.
cols = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
key S = sign | c1
key E = exponent | c3
key M = mantissa | c2
key I = integer bits | wine
FP16 | S E E E E E M M M M M M M M M M
BF16 | S E E E E E E E E M M M M M M M
FP8 E4M3 | S E E E E M M M . . . . . . . .
FP8 E5M2 | S E E E E E M M . . . . . . . .
INT8 | I I I I I I I I . . . . . . . .
INT4 | I I I I . . . . . . . . . . . .
%%
~~~

| Format | Largest value | Relative precision | Typical use |
| --- | --- | --- | --- |
| BF16 | about ⟦m632⟧ | 0.39% | Default weights and activations |
| FP16 | 65,504 | 0.05% | Older GPUs, needs care with overflow |
| FP8 E4M3 | 448 | 6.25% | Weights, activations, KV |
| FP8 E5M2 | 57,344 | 12.5% | Gradients; rarely for inference |
| INT8 | 127 × scale | uniform steps | Weights, activations with scales |
| INT4 | 7 × scale | uniform steps | Weights only, with group scales |

FP8 E4M3 is only accurate to about 6%, so it always travels with a scale that moves each tensor's values into its sweet spot.

<div class="take">
<div class="lab">Key takeaway</div>

quantization cuts bytes and, with hardware support, the cost of each FLOP. Match the format to the phase you are bound by and to what your GPU's tensor cores can actually compute.

</div>
