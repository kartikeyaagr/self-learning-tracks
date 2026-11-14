---
id: "13.06"
course: "inference-eng-course"
module: "13"
slug: "write-rope-fused-operators"
title: "Write RoPE & Fused Operators"
keyIdeas: ["RoPE rotates pairs of dimensions by position-dependent angles.", "It is elementwise given cos and sin tables: cheap math, memory bound.", "Fuse it with neighboring ops: QKV split, RoPE, and KV cache write in one kernel.", "Other common fusions: SiLU x up (SwiGLU), add + norm, dequant + GEMM.", "Every fusion removes at least one HBM round trip and one launch."]
question: "What does rotary position embedding do to Q and K, and why is it a good candidate for fusion with the QKV projection output?"
answer: "RoPE rotates each pair of dimensions of Q and K by an angle proportional to the token position, so dot products depend on relative position. It is purely elementwise on the projection output, so applying it right after the QKV GEMM (or while writing K into the KV cache) avoids an extra HBM read and write of Q and K."
sources: [{"title": "RoFormer: Enhanced Transformer with Rotary Position Embedding", "url": "https://arxiv.org/abs/2104.09864", "note": "RoPE paper."}, {"title": "Triton tutorials", "url": "https://triton-lang.org/main/getting-started/tutorials/index.html", "note": "Kernel patterns to build on."}]
tags: ["triton", "rope"]
---

## RoPE in brief

Rotary position embeddings (RoPE) encode each token's position by rotating pairs of dimensions in its query and key vectors. For a dimension pair (x1, x2) at position p, with a per-pair frequency f_i:

```
theta = p * f_i
x1_new = x1 * cos(theta) - x2 * sin(theta)
x2_new = x1 * sin(theta) + x2 * cos(theta)
```

The frequencies fall geometrically across the pairs, f_i = base^(−2i ÷ head_dim), with a base such as 10,000 or 500,000. RoPE is applied to Q and K, not to V. Because both are rotated by their own positions, the dot product of a query and a key depends only on the difference between their positions, which is what the model learned to use.

## The cost of doing it unfused

In a plain implementation, the QKV projection writes Q and K to memory, a RoPE kernel reads them, rotates them and writes them back, and a cache-write kernel reads K and V again to copy them into the paged cache. That is three passes over K and two over Q, plus three launches per layer.

## A fused kernel

One program per (token, head) can do all of it:

1. Load the head's slice of Q or K for this token, straight from the projection output.
2. Load cos and sin for the token's position from a precomputed table, or compute them.
3. Rotate the pairs in registers.
4. Store Q where attention will read it.
5. Store K directly into the paged KV cache at the token's slot, given by the slot mapping from the scheduler. Store V there too.

Three passes and three launches become one.

<div class="ex">
<div class="lab">Worked example</div>

**what fusion saves per step**

A decode step at batch 128 on Llama 3 8B: per layer, Q is 128 × 4,096 values and K and V are 128 × 1,024 each, in BF16, about 1.5 MB in total.

Unfused: about 4.5 MB of traffic and 3 launches per layer, about 15 µs. Fused: about 1.5 MB and 1 launch, about 4 µs.

Over 32 layers that saves about 0.35 ms per step, a few percent of the step. Small on its own, but fusions like this add up.

</div>

## Other fusions worth knowing

| Fusion | What it saves |
| --- | --- |
| SiLU(gate) × up | writing and reading the intermediate activation |
| Residual add + RMSNorm | one pass over the hidden state |
| RoPE + KV cache write | passes over Q and K, two launches |
| Dequantize + matmul | a full-precision copy of the weights |
| Bias and activation in the matmul epilogue | one pass over the output |
| RMSNorm + FP8 activation quantization | one pass, and a separate quantize kernel |
| Softmax + top-k + sampling | passes over large logit tensors |

The rule for finding fusions: look for any tensor that is written to HBM only to be read straight back by the next kernel.

## Get the variant exactly right

RoPE has several variants, and using the wrong one gives fluent but wrong outputs, which are hard to catch:

- **Interleaved pairs**, where dimensions (0, 1), (2, 3) and so on are paired, versus **split halves**, where dimension i is paired with i + head_dim ÷ 2. Different model families use different layouts.
- **Partial rotary**, where only some of each head's dimensions are rotated.
- **Long-context scaling** methods, such as linear position interpolation, NTK-aware scaling, YaRN, or the frequency-dependent scaling used by Llama 3.1. They change the frequencies, sometimes per dimension.
- **Multimodal variants**, which use several position components for images or video.

Always test against the model's reference implementation on real checkpoints and at long positions, not only near position 0, where every variant looks similar.

## Deep dive: RoPE's wavelengths

RoPE rotates dimension pair ⟦m743⟧ by angle ⟦m744⟧ at position ⟦m745⟧, with ⟦m746⟧. Pair ⟦m747⟧ repeats every ⟦m748⟧ positions:

⟦M741⟧

~~~dg-raw
%%chart Wavelength of each dimension pair for Llama 3 (β = 500,000, d_h = 128).
x = 0, 16, 32, 48, 63
logy = 1
xlab = Pair index i
ylab = Wavelength (tokens)
Wavelength: 6.3, 167, 4443, 118000, 2560000
%%
~~~

Early pairs turn quickly and encode fine position. Late pairs barely turn across the whole context and act almost like position-free channels. Because a rotation by ⟦m749⟧ followed by the dot product with a rotation by ⟦m750⟧ depends only on ⟦m751⟧, attention sees relative positions:

⟦M742⟧

A fused RoPE kernel must reproduce the exact pairing convention (adjacent pairs or split halves) and ⟦m752⟧ of the checkpoint. Getting either wrong gives fluent but subtly worse output, not an error.

<div class="take">
<div class="lab">Key takeaway</div>

RoPE is cheap elementwise math that becomes nearly free when fused with the QKV output and the KV cache write. Look for fusions wherever a tensor goes to HBM only to be read straight back, and match the model's exact RoPE variant.

</div>
