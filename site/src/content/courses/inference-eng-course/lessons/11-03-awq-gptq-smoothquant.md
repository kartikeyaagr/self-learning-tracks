---
id: "11.03"
course: "inference-eng-course"
module: "11"
slug: "awq-gptq-smoothquant"
title: "AWQ, GPTQ & SmoothQuant"
keyIdeas: ["All three are post-training methods using a small calibration set.", "GPTQ: quantize column by column, update remaining weights to cancel error.", "AWQ: protect salient channels by scaling them before quantization.", "SmoothQuant: shift activation outliers into weights to enable W8A8.", "Calibration data should resemble your traffic."]
question: "In one sentence each, what problem do GPTQ, AWQ and SmoothQuant solve?"
answer: "GPTQ quantizes weights layer by layer, adjusting remaining weights to compensate for rounding error using second-order information from calibration data. AWQ finds the small fraction of weight channels that matter most (by activation magnitude) and scales them up before quantizing so they lose less precision. SmoothQuant moves outlier magnitude from activations into weights with a per-channel scale, so both can be quantized to 8 bits."
sources: [{"title": "GPTQ", "url": "https://arxiv.org/abs/2210.17323", "note": "Accurate post-training quantization."}, {"title": "AWQ", "url": "https://arxiv.org/abs/2306.00978", "note": "Activation-aware weight quantization."}, {"title": "SmoothQuant", "url": "https://arxiv.org/abs/2211.10438", "note": "Smoothing activation outliers for W8A8."}]
tags: ["quantization"]
---

## Why naive rounding is not enough at 4 bits

Rounding each weight to the nearest level minimizes the error in each weight on its own. But what matters is the error in the layer's output, which depends on the inputs the weights are multiplied by. Some weights are multiplied by large activations and matter a lot; others barely affect the output. The three methods in this lesson use real activations from a small calibration set to decide how to quantize.

## GPTQ: compensate for rounding error

GPTQ quantizes one layer at a time, one column of weights at a time.

1. Run a few hundred calibration samples through the model and record each layer's inputs.
2. Quantize one column of the weight matrix.
3. Measure how much that rounding changes the layer's output on the calibration inputs.
4. Adjust the remaining, not yet quantized columns to cancel out that change as far as possible.
5. Repeat for the next column.

The adjustment uses an approximation of second-order information, the Hessian of the layer's output error, computed from the calibration inputs. The result is that errors from early columns are absorbed by later ones instead of adding up. GPTQ produces good INT4 and usable INT3 weights, and quantizing a 70B model takes a few hours on one GPU.

## AWQ: protect the important channels

AWQ (activation-aware weight quantization) starts from an observation: about 1% of weight channels matter far more than the rest. They are the channels that multiply large activation values. Rounding errors there are amplified.

Keeping those channels in high precision would need a mixed-precision kernel. Instead, AWQ:

1. Measures the average activation magnitude per input channel on calibration data.
2. Multiplies the important weight channels by a scale greater than 1 before quantizing, so they use more of the available levels and get finer effective precision.
3. Divides the matching activations by the same scale, folding that division into the previous layer's operation, so the product is unchanged.

AWQ searches for the scales that minimize output error. It is fast, needs little calibration data, and generalizes well across tasks, which made it a popular choice for INT4 checkpoints.

## SmoothQuant: move the difficulty from activations to weights

SmoothQuant targets W8A8, where activations are also quantized. The problem: LLM activations have a few outlier channels with values tens of times larger than the rest. Per-tensor INT8 on such activations loses most of their precision.

The trick is a mathematically equivalent rewrite. For a per-channel scale s:

X · W = (X ÷ s) · (s · W)

Dividing the activations by s shrinks the outlier channels. Multiplying the matching weight rows by s makes the weights a little harder to quantize, but weights tolerate it much better. A migration strength parameter, often around 0.5, balances how much difficulty moves. Then both are quantized to INT8 and run on INT8 tensor cores.

## Comparison

| Method | Quantizes | Typical bits | Main idea | Calibration |
| --- | --- | --- | --- | --- |
| GPTQ | weights | 4, sometimes 3 | compensate rounding error in later columns | a few hundred samples, hours for large models |
| AWQ | weights | 4 | scale up salient channels before rounding | small set, fast |
| SmoothQuant | weights and activations | 8 | migrate activation outliers into weights | small set, fast |
| FP8 (dynamic or static scales) | weights and activations | 8 | wide-range float format | little or none |

<div class="ex">
<div class="lab">Worked example</div>

**choosing a method for a deployment**

A 70B model must serve on two 80 GB GPUs with room for KV cache.

BF16 weights need 140 GB: too much. FP8 needs 70 GB: fits, with about 70 GB for KV across both GPUs. INT4 AWQ needs about 38 GB: fits easily, with about 100 GB for KV.

On H100s, FP8 W8A8 is the first choice: near-lossless and fast at all batch sizes. If the eval shows FP8 is fine but KV capacity is still short, INT4 AWQ weights with FP8 KV are the next option, accepting some prefill slowdown at large batch.

</div>

## Practical notes

- **Use pre-quantized checkpoints when they exist.** Many popular models have official or well-tested FP8, AWQ and GPTQ versions. Check that your engine supports the exact format.
- **Calibrate on data like your traffic.** A code model calibrated on English prose can lose more quality than necessary. A few hundred samples from your domain are usually enough.
- **Kernel support decides real speed.** INT4 weight-only checkpoints are fast only if the engine has an efficient mixed-precision kernel for your GPU, such as Marlin or Machete on NVIDIA. Without one, the kernel can be slower than BF16.
- **Some layers are often left unquantized**, such as the LM head and embeddings, because they are sensitive and small relative to the rest.

## Deep dive: the objectives of GPTQ, AWQ and SmoothQuant

All three minimize the error in a layer's output, not in its weights. For inputs ⟦m645⟧ collected on calibration data:

⟦M641⟧

**GPTQ** quantizes one column at a time and pushes each column's rounding error onto the columns not yet quantized, using the inverse Hessian ⟦m646⟧ with ⟦m647⟧:

⟦M642⟧

The update is exact for a quadratic loss. It lets later columns absorb earlier errors, so the layer's output stays close even though every weight moved.

**AWQ** scales input channel ⟦m648⟧ by ⟦m649⟧ before quantizing and divides the activation by the same amount, which leaves the math unchanged:

⟦M643⟧

where ⟦m650⟧ is the mean size of channel ⟦m651⟧'s activations. Channels that see large activations get finer quantization. ⟦m652⟧ is chosen by a small grid search on calibration data.

**SmoothQuant** moves outliers from activations, which are hard to quantize, into weights, which are easy:

⟦M644⟧

With ⟦m653⟧, both sides end up with similar crest factors, and both can be INT8.

<div class="take">
<div class="lab">Key takeaway</div>

GPTQ compensates for rounding error, AWQ protects the channels that matter most, and SmoothQuant moves activation outliers into the weights. Choose by bit width, hardware and kernel support, and calibrate on data like your traffic.

</div>
