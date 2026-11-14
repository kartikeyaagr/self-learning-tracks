---
id: "11.02"
course: "inference-eng-course"
module: "11"
slug: "bf16-fp8-int8-4-bit-inference"
title: "BF16, FP8, INT8 & 4-Bit Inference"
keyIdeas: ["Quantize: q = round(x / scale); dequantize: x = q x scale.", "Scale granularity: per tensor, per channel, per group, per token (activations).", "FP8 has an exponent, so it handles range better than INT8 at the same bits.", "4-bit needs fine-grained group scales to keep accuracy.", "Outlier channels in activations are the main obstacle to 8-bit activations."]
question: "Compare per-tensor, per-channel and per-group scaling for weight quantization, and say which INT4 methods usually use."
answer: "Per-tensor uses one scale for the whole matrix: cheap but outliers force a coarse scale for everything. Per-channel uses one scale per output channel: better accuracy, still cheap. Per-group uses one scale per small group (e.g. 128 weights) along the input dimension: best accuracy at low bits but more scale storage and dequant work. INT4 methods like GPTQ and AWQ typically use per-group scaling with group size 128."
sources: [{"title": "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", "url": "https://arxiv.org/abs/2208.07339", "note": "Outliers and mixed precision."}, {"title": "FP8 Formats for Deep Learning", "url": "https://arxiv.org/abs/2209.05433", "note": "E4M3 and E5M2 definitions."}]
tags: ["quantization"]
---

## The basic operation

Quantization maps real values onto a small set of levels using a scale:

```
scale = max(|x|) / q_max          # q_max = 127 for INT8, 7 for INT4
q = round(x / scale)              # store q in low precision
x_hat = q * scale                 # dequantize when needed
```

The error x − x_hat comes from two sources:

- **Rounding.** Each value moves to the nearest level. With 256 levels the error is small. With 16 levels it is much larger.
- **Outliers.** One very large value sets the scale for the whole group. Every other value then uses only a few of the available levels, and many small values round to zero.

~~~dg-raw
%%widget quant Drop to 4 bits with a per-tensor scale and watch most weights round to zero. Then switch to group scales.
bits = 4
%%
~~~

## Granularity: how many values share a scale

| Granularity | One scale per | Accuracy | Overhead |
| --- | --- | --- | --- |
| Per tensor | whole matrix | lowest | none |
| Per channel | output row of the weight | good | tiny |
| Per group (often 128) | block of 128 input values | best at 4 bits | scales add about 3% at 4 bits |
| Per token | row of activations | good for activations | computed at runtime |
| Per block (MX formats) | 16 or 32 values | very good | built into the format |

Smaller groups mean each scale fits its values better, so outliers affect fewer neighbours. The cost is storing more scales and more work in the kernel.

<div class="ex">
<div class="lab">Worked example</div>

**what group scales do**

A row of 4,096 weights has typical values around ±0.02 and one outlier of 0.5.

INT4 per row: scale = 0.5 ÷ 7 ≈ 0.071. A typical weight of 0.02 rounds to 0.02 ÷ 0.071 ≈ 0.28, which becomes 0. Most of the row is lost.

INT4 with groups of 128: only the group containing the outlier gets the large scale. In the other 31 groups, the scale is about 0.03 ÷ 7 ≈ 0.0043, and a 0.02 weight becomes level 5, accurate to within about 10%.

Storage: one 16-bit scale per 128 weights adds 0.125 bits per weight, so INT4 costs about 4.125 bits per weight.

</div>

## FP8 compared with INT8

FP8 E4M3 has 4 exponent bits and 3 mantissa bits. Its levels are dense near zero and sparse for large values, so small and large values keep similar relative precision. INT8 levels are evenly spaced, so a large outlier wastes most of them.

In practice, FP8 with a per-tensor or per-channel scale often matches BF16 quality closely on weights and activations. That is why FP8 W8A8 is a common default on H100-class GPUs. INT8 W8A8 needs techniques such as SmoothQuant to handle activation outliers.

## 4-bit weights

Uniform 4-bit has only 16 levels, so naive rounding loses too much. Two things make 4-bit work:

- **Group scales**, and sometimes zero points (an offset for asymmetric ranges).
- **Calibration-based methods**, such as GPTQ and AWQ, that choose quantized values to minimize the error in the layer's outputs, not just in the weights. The next lesson covers them.

The newer FP4 formats on Blackwell use small blocks (16 or 32 values) with their own scales, so they keep accuracy closer to FP8 than plain INT4 does, while running natively on the tensor cores.

## Typical quality impact

| Format | Typical quality compared with BF16 |
| --- | --- |
| FP8 W8A8 | near lossless on most tasks |
| INT8 W8A8 with SmoothQuant | small loss |
| INT4 W4A16 with AWQ or GPTQ | small to moderate loss, task dependent |
| FP4 with block scales | small loss with good recipes, still maturing |

Treat these as starting expectations. Larger models usually tolerate quantization better than smaller ones. Math, code and long-context tasks usually show losses first. Your own evaluation decides.

## Deep dive: rounding error and signal-to-noise

Symmetric quantization with ⟦m636⟧ bits uses ⟦m637⟧:

⟦M633⟧

Error is set by the scale, and the scale is set by the largest value. With crest factor ⟦m638⟧, the signal-to-quantization-noise ratio is about

⟦M634⟧

Each extra bit buys 6 dB. Each 10-fold outlier costs 20 dB, more than three bits.

~~~dg-raw
%%chart SQNR against bits, for a well-behaved group (C = 4) and a group with one strong outlier (C = 25).
x = 3, 4, 5, 6, 8
xlab = Bits
ylab = SQNR (dB)
Group without outlier (C = 4): 10.8, 16.8, 22.8, 28.8, 40.9
Group with outlier (C = 25): -5.1, 0.9, 6.9, 12.9, 25.0
%%
~~~

At 4 bits, a group with an outlier has SQNR near 0 dB: the noise is as large as the signal. That is why small groups (32 to 128 values) are essential at 4 bits: they confine the large scale to the one group that needs it.

## Deep dive: asymmetric quantization

When values are not centered on zero, a zero point ⟦m639⟧ uses all levels:

⟦M635⟧

It is common for INT4 weights (with ⟦m640⟧ stored per group) and for activations after a ReLU-like function. It costs one extra subtraction in the kernel.

<div class="take">
<div class="lab">Key takeaway</div>

precision loss comes from rounding and from outliers that stretch the scale. Finer-grained scales and floating-point formats fight both. 8 bits are usually safe, while 4 bits need group scales, good methods and careful evaluation.

</div>
