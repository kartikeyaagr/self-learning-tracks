---
id: "07.01"
course: "ai-engineering-course"
module: "07"
slug: "quantization"
title: "Quantization"
keyIdeas: ["Quantization stores weights (and sometimes activations and KV cache) in fewer bits: fp16 to int8, int4 or fp8.", "Memory for weights = parameters x bits / 8. 70B in fp16 is 140 GB; int8 is 70 GB; 4-bit is about 35 GB.", "Decode is memory-bandwidth bound, so fewer bytes per weight also means faster tokens.", "Basic method: scale a block of values by its max (absmax) or min/max range to fit the integer range; store the scale.", "Outliers break naive quantization. GPTQ uses second-order information to correct errors; AWQ protects weights tied to large activations.", "Post-training quantization (PTQ) needs no retraining; quantization-aware training (QAT) trains with simulated low precision. Quality loss is small at 8-bit and grows below 4-bit."]
question: "A 70B model in fp16 needs how much memory for weights? What about int8 and 4-bit, and what do you risk at 4-bit?"
answer: "Weights = params x bytes: 70B x 2 bytes = 140 GB in fp16; 70B x 1 byte = 70 GB in int8; 70B x 0.5 byte = ~35 GB at 4-bit (a bit more for per-block scales). At 4-bit, rounding error and outliers can hurt quality, mostly on maths, code, long-context and rare languages, so you'd use outlier-aware methods (GPTQ, AWQ) and test on your own evals."
sources: [{"title": "Maarten Grootendorst: A Visual Guide to Quantization", "url": "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization", "note": "Number formats, absmax, GPTQ, AWQ and GGUF with diagrams."}, {"title": "Frantar et al.: GPTQ (2022)", "url": "https://arxiv.org/abs/2210.17323", "note": "Post-training 3-4 bit quantization using second-order information."}, {"title": "Lin et al.: AWQ (2023)", "url": "https://arxiv.org/abs/2306.00978", "note": "Protect the weights that matter for large activations."}]
---

## Fewer bits per weight

A model's weights are numbers, usually stored as 16-bit floats (fp16 or bf16). Quantization stores them with fewer bits. Grootendorst's visual guide is the best single introduction.

| Format | Bits | 70B model weights |
| --- | --- | --- |
| fp32 | 32 | 280 GB |
| fp16 / bf16 | 16 | 140 GB |
| int8 / fp8 | 8 | 70 GB |
| int4 / 4-bit | 4 | ~35 GB (plus scales) |

Two benefits:

- **Fits on fewer GPUs**: a 70B model at 4-bit fits on one 48 GB GPU or two 24 GB ones.
- **Faster decode**: decode reads every weight per token and is memory-bound, so half the bytes is close to twice the tokens per second, if the kernels are good.

## How a value gets quantized

The simplest scheme is **absmax**: for a block of weights, find the largest absolute value, scale so it maps to 127, round each value to the nearest integer.

```
scale = 127 / max(|w|)
q = round(w x scale)        # int8
w_restored = q / scale
```

The rounding error is the quality cost. Store one scale per block (say 64 or 128 weights) instead of per tensor, so one large value doesn't wreck the precision of everything else.

## The outlier problem

LLM activations and some weights have rare, very large outliers. With absmax, a single outlier stretches the scale and every normal value gets squeezed into a few integer levels. Methods to deal with it:

- **LLM.int8()**: keep outlier dimensions in fp16, quantize the rest.
- **GPTQ** (Frantar et al.): quantize weights one column at a time and adjust the remaining weights to compensate for the error, using second-order (Hessian) information from a small calibration set.
- **AWQ** (Lin et al.): find the ~1% of weight channels that multiply large activations and scale them to protect them before quantizing.
- **GGUF k-quants** (llama.cpp): block-wise mixed-precision formats for CPU and laptop inference.

## PTQ vs QAT

~~~dg-flow Two ways to get a quantized model.
[Post-training] Trained fp16 model -(calibrate on a small dataset)-> *Quantized model
[Quantization-aware] Model -(train with simulated low precision)-> Quantized model that expects the noise
~~~

Post-training quantization is fast and needs no training data beyond calibration. QAT is more work but recovers more quality at very low bits.

## What you risk

At 8 bits the loss is usually negligible. At 4 bits most chat benchmarks hold up well, but losses show up first in maths, code, long-context recall and less common languages. Below 4 bits quality drops quickly for most models. Always evaluate on your own task.

## Quantize a row of weights

The figure quantizes a row of weights to the chosen bit width. Add an outlier, change the block size that shares one scale, and watch the rounding error.

~~~dg-raw
%%widget quant Rounding weights to fewer bits, and how one outlier stretches the scale for its whole block.
~~~

The basic scheme, symmetric absmax quantization:

```
scale = max(|w|) / (2^(bits-1) - 1)        # 127 for int8, 7 for int4
q     = round(w / scale)                    # stored as a small integer
w_hat = q * scale                           # used at inference
```

With int4 there are only 15 usable levels. If one weight is 20× larger than the rest, the scale is set by it, and most normal weights round to 0 or ±1. Smaller blocks (32 to 128 weights per scale) contain the damage, at the cost of storing more scales.

## Methods you will see

| Method | What it does | Typical use |
| --- | --- | --- |
| LLM.int8() | Keeps outlier feature dimensions in fp16, the rest in int8 | 8-bit inference; showed outliers appear around 6.7B parameters |
| GPTQ | Quantizes column by column, adjusting remaining weights to cancel the error, using a small calibration set | 4-bit GPU inference |
| AWQ | Finds the weights that matter most for activations and protects them with per-channel scaling | 4-bit GPU inference |
| GGUF k-quants | Block-wise mixed precision formats in llama.cpp | CPU and laptop inference |
| FP8 (E4M3) | 8-bit floating point, supported in hardware on H100 and newer | Fast serving, sometimes training |

## Measure, don't assume

Perplexity on general text often barely moves at 4 bits while specific skills drop: long multi-step math, code, low-resource languages and long-context recall tend to suffer first. Run your own eval set on the quantized model before switching, and compare at the task level.

<div class="ex">
<div class="lab">Worked example</div>

**what 4-bit buys you**

A 70B model needs about 140 GB in bf16 and about 40 GB at 4 bits with group scales.

That moves it from two 80 GB GPUs to one, and since decode is memory-bound, fewer bytes per weight also means faster tokens.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

weight memory = parameters x bits / 8. Quantization shrinks it and speeds up memory-bound decode; outlier-aware methods like GPTQ and AWQ keep 4-bit models usable, but test the tasks you care about.

</div>
