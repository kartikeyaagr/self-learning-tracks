---
id: "13.04"
course: "ai-research-track-web"
module: "13"
slug: "mixed-precision-and-activation-checkpointing"
title: "Mixed precision and activation checkpointing"
keyIdeas: ["A float format splits its bits between **exponent** (range) and **mantissa** (precision). bf16 keeps fp32's range with 8 significant bits; fp16 has more precision but overflows above 65504; fp8 comes as E4M3 (max 448) and E5M2 (max 57344).", "Matmuls run in 16 or 8 bits, but the optimiser updates **fp32 master weights**. A bf16 weight ignores any update smaller than half the gap to its neighbour (0.0039 at 1.0), so pure-bf16 training silently stalls late in a decaying schedule.", "fp16 needs **loss scaling** to keep small gradients from underflowing; bf16 does not. fp8 needs a **scale factor** per tensor or, more robustly, per small block, because one outlier channel can push every other value toward zero.", "**Activation checkpointing** keeps only each layer's input and recomputes the rest in backward. Full recomputation costs about one extra forward pass, raising training FLOPs from about $6N$ to $8N$ per token.", "**Selective recomputation** (and FlashAttention) recomputes only the attention core, which is memory-heavy but cheap, removing the quadratic activation term for a few percent of extra compute. Try it before full checkpointing."]
question: "To save memory a teammate stores weights and Adam moments in bf16, with no fp32 master copy. The learning rate decays to $3 \\times 10^{-5}$ by the end of training and typical weights have magnitude around 0.02. Adam's per-step updates are roughly the learning rate in size. What happens to the updates, and what would you change? Separately, the same run uses full activation checkpointing on every layer; by how much does that inflate the FLOPs per token, what happens to the reported MFU, and what would you try first instead?"
answer: "A bf16 weight of about 0.02 lies in the band $[2^{-6}, 2^{-5})$ = [0.0156, 0.0313), where consecutive bf16 values are $2^{-6} \\times 2^{-7} = 2^{-13} \\approx 1.22 \\times 10^{-4}$ apart. An update of $3 \\times 10^{-5}$ is below half that gap ($6.1 \\times 10^{-5}$), so it is rounded away: the weight does not move, and late training stalls while the loss curve flattens for no visible reason. Larger updates are quantised to multiples of the gap, so even those that survive are distorted (an update of $10^{-4}$ becomes a step of $1.22 \\times 10^{-4}$, 22 percent too big). The fix is fp32 master weights (4 extra bytes per parameter, usually recovered by sharding them with ZeRO), or if memory forbids it, stochastic rounding or Kahan-compensated updates in bf16. Keep both Adam moments in fp32 too: the second moment holds squared gradients, which underflow or lose precision easily in 16 bits, and it is used in a square root and a division. A useful diagnostic is the per-layer ratio $|\\Delta w|/|w|$: below about $2^{-9}$ to $2^{-8}$ with bf16 storage, updates are being lost. On recomputation: full checkpointing adds one forward pass, about $2N$ on top of $6N$, so 33 percent more FLOPs per token ($8N/6N$). MFU counts only the $6N$ model FLOPs, so it falls to about 75 percent of its previous value even if the GPUs are equally busy (HFU, which counts recomputation, would not fall). First try selective recomputation of the attention core, or simply FlashAttention, which removes the quadratic $5as^2b$ activation term at a few percent cost, plus sequence parallelism if tensor parallelism is in use; fall back to full checkpointing only for the layers or pipeline stages that still do not fit."
sources: [{"title": "Micikevicius et al.: Mixed Precision Training (2017)", "url": "https://arxiv.org/abs/1710.03740", "note": "The three ingredients: fp32 master weights, loss scaling and fp32 accumulation, with the evidence for each."}, {"title": "Micikevicius et al.: FP8 Formats for Deep Learning (2022)", "url": "https://arxiv.org/abs/2209.05433", "note": "The E4M3 and E5M2 definitions, their ranges and how scaling is applied in training."}, {"title": "Korthikanti et al.: Reducing Activation Recomputation in Large Transformer Models (2022)", "url": "https://arxiv.org/abs/2205.05198", "note": "Selective recomputation and sequence parallelism, with the memory and overhead measurements."}, {"title": "Chen et al.: Training Deep Nets with Sublinear Memory Cost (2016)", "url": "https://arxiv.org/abs/1604.06174", "note": "The original gradient-checkpointing analysis and the square-root schedule."}]
practice: ["mixed-precision-training-memory-accountant", "mixed-precision-with-loss-scaling", "rematerialized-mlp-block", "block-wise-fp8-quantization"]
tags: ["mixed precision", "fp8", "activation checkpointing"]
tools: ["PyTorch", "Transformer Engine", "Megatron-LM"]
---

## What you will learn

- How exponent bits (range) and mantissa bits (precision) differ across fp32, bf16, fp16 and fp8.
- Why training computes in 16 or 8 bits but updates an fp32 **master copy** of the weights.
- Why fp16 needs **loss scaling**, bf16 does not, and fp8 needs a **scale factor** per tensor or block.
- How **activation checkpointing** trades one extra forward pass for most activation memory, and how **selective recomputation** gets most of that saving almost for free.
- How to check that a low-precision or recomputing run is behaving.

*Before you start:* you need the memory and FLOP accounting of lesson 13.01 (the 16-byte rule, the $6N$ rule and the $34sbh$ activation formula). The PyTorch course covered `autocast` and `GradScaler`; here we look at the numerics underneath them.

## Precision and memory are both budgets

Lessons 13.02 and 13.03 split state and layers across GPUs, but each GPU still holds its micro-batch's activations, and each matmul runs at the speed its number format allows. This lesson attacks both with one idea: spend a cheap resource to save an expensive one. Lower precision spends numerical accuracy to buy speed and memory; recomputation spends spare FLOPs to buy memory.

The payoff is large. On the H100 SXM spec sheet, dense tensor-core throughput is roughly 495 TFLOPS in tf32, 989 in bf16 and 1979 in fp8 (spec sheet values quoted from memory), and every halving of bytes per value halves memory traffic and activation storage.

## The idea in one picture

A floating-point number is stored as a sign, an **exponent** (which power of two the number sits near) and a **mantissa** (the significant digits). Exponent bits decide the **range**: how large and how small a number can be. Mantissa bits decide the **precision**: how close together neighbouring representable numbers are.

| format | exponent / mantissa bits | largest finite | smallest normal | gap just above 1.0 | typical role
| --- | --- | --- | --- | --- | --- |
| fp32 | 8 / 23 | about $3.4 \times 10^{38}$ | about $1.2 \times 10^{-38}$ | $2^{-23} \approx 1.2 \times 10^{-7}$ | master weights, optimiser moments, reductions
| bf16 | 8 / 7 | about $3.4 \times 10^{38}$ | about $1.2 \times 10^{-38}$ | $2^{-7} \approx 0.0078$ | matmul inputs, activations, gradients
| fp16 | 5 / 10 | 65504 | about $6.1 \times 10^{-5}$ | $2^{-10} \approx 0.001$ | the same, on older GPUs, with loss scaling
| fp8 E4M3 | 4 / 3 | 448 | $2^{-6} \approx 0.016$ | $2^{-3} = 0.125$ | forward weights and activations
| fp8 E5M2 | 5 / 2 | 57344 | about $6.1 \times 10^{-5}$ | $2^{-2} = 0.25$ | gradients, which need range more than precision

The figure enumerates every positive number each 16- and 8-bit format can represent and plots the gap to the next one, relative to the number.

~~~dg-fig 13.04-formats.svg
Height is precision (lower is finer), horizontal extent is range. bf16 matches fp32's range but is about 65,000 times coarser; fp16 is 8 times finer than bf16 but stops at 65,504; the fp8 formats cover only a narrow window. The sawtooth comes from the gap doubling at each power of two.
~~~

bf16 has fp32's reach but only about 3 significant decimal digits. fp16 is finer but stops at 65,504 and fades into **subnormals** (numbers below the smallest normal, with ever fewer significant bits) on the left. Every recipe below puts each tensor in a format whose window and spacing fit its values.

## Step 1: compute low, accumulate high

- **Matmuls** multiply and sum millions of numbers. Coarse inputs are fine: the errors are small and random, and the tensor core accumulates the sum at higher precision.
- **Accumulations over time** (an update added to a weight, a squared gradient added into Adam's second moment) add small numbers to a large running value, step after step. If the addition rounds away every time, the error is the whole update.

So mixed-precision training (Micikevicius et al.) keeps fp32 **master weights**, casts them to 16 bits for forward and backward, and applies the optimiser update to the fp32 copy.

~~~dg-flow One mixed-precision step: low precision where it is fast, fp32 where it accumulates.
*fp32 master weights -(cast)-> bf16 weights -> Forward and backward in bf16 -> bf16 gradients -(reduce, upcast)-> fp32 Adam update -> *fp32 master weights
~~~

Softmax, norms, the loss and gradient all-reduces also sum many small terms, so they run in fp32. The master copy is 4 of lesson 13.01's 16 bytes per parameter.

## Step 2: why small updates vanish

Take a weight $w$ stored in a format with $m$ mantissa bits. Within one power-of-two band, $2^e \le |w| < 2^{e+1}$, neighbouring representable numbers are

$$\text{gap}(w) = 2^{e} \times 2^{-m}$$

apart, where $2^e$ is the largest power of two not above $|w|$ and $m$ is 7 for bf16 or 23 for fp32. The sum $w + u$ is rounded to the nearest representable number, so any update with $|u| < \text{gap}/2$ disappears.

<div class="ex">
<div class="lab">Worked example</div>

**a thousand updates that go nowhere**

A weight starts at 1.0 and receives 1,000 updates of $+10^{-4}$. The right answer is $1.0 + 1000 \times 10^{-4} = 1.1$.

In fp32 the gap at 1.0 is $2^{-23} \approx 1.19 \times 10^{-7}$, about 840 times smaller than the update, so each addition is almost exact: the run ends at 1.1000166.

In bf16, $e = 0$ and the gap is $2^{-7} = 0.0078125$, half of which is $0.0039$. The update is 39 times smaller than that, so $1 + 10^{-4}$ rounds back to 1.0, every time. After 1,000 steps the weight is still exactly 1.0.

</div>

Adam's per-step update is roughly the learning rate in size, and late in a decay schedule that falls below the half-gap for most weights. Updates that do survive are snapped to whole gaps, so they are distorted too. The run does not crash; it quietly stops learning.

~~~dg-fig 13.04-stall.svg
Left: the hand example run for real. bf16 accumulation (dashed) never moves; fp32 gives the straight line to 1.1; stochastic rounding moves in coarse steps but tracks it on average. Right: a 32-weight regression trained with Adam and cosine decay. Both runs compute in bf16; only the one without an fp32 master copy stalls, at a loss about 1,000 times higher.
~~~

In the right panel (learning rate decaying from $10^{-2}$ to $10^{-4}$ over 3,000 steps) fp32 masters reach a training MSE of $1.4 \times 10^{-4}$; bf16 weights stall at $0.14$, and over the last 500 steps 97 percent of individual weight updates were rounded away.

Alternatives exist: **stochastic rounding** rounds up or down at random in proportion to the distance, so updates survive on average, and **Kahan summation** keeps a buffer of the rounding error and feeds it back. fp32 masters remain the default because they are simple and robust, and ZeRO (13.02) shards them anyway.

## Step 3: loss scaling for fp16

fp16 has the opposite problem: fine precision, short range. Its smallest normal is $2^{-14} \approx 6.1 \times 10^{-5}$, subnormals reach $2^{-24} \approx 6.0 \times 10^{-8}$, and anything smaller becomes 0. Gradients are often that tiny, because the loss averages over many tokens.

~~~dg-fig 13.04-underflow.svg
Left: activation-gradient magnitudes in a 7-layer MLP trained briefly on the digits data. Most land in fp16's subnormal zone and 8 percent below it. Right: multiplying by a scale S before casting rescues them; 2^8 already cuts the loss to 0.4 percent, and overflow only starts around 2^25.
~~~

The fix is **loss scaling**: multiply the loss by a constant $S$ before backward, so by the chain rule every gradient is multiplied by $S$ and lifted into range, then divide by $S$ in fp32 before the update:

$$g_{\text{true}} = \frac{1}{S}\,\nabla_\theta\big(S \cdot \mathcal{L}\big).$$

Here $\mathcal{L}$ is the loss and $\theta$ the parameters; $S$ is a power of two, so scaling is exact.

The safe window in the figure is wide (about $2^{8}$ to $2^{24}$) but it moves during training, so **dynamic loss scaling** tracks it: start large (PyTorch's `GradScaler` uses $2^{16}$), halve $S$ and skip the step whenever an inf or NaN appears, and double it after a run of clean steps (2,000 by default).

bf16 shares fp32's 8 exponent bits, so nothing in the figure would underflow. That is why bf16 replaced fp16 for LLM training wherever hardware supports it: no scaler, no skipped steps.

<div class="ex">
<div class="lab">Worked example</div>

**one gradient, with and without scaling**

A gradient $g = 10^{-8}$ is below half of fp16's smallest subnormal ($2^{-24}/2 \approx 2.98 \times 10^{-8}$), so the cast gives exactly 0.

With $S = 1024$ it becomes $1.024 \times 10^{-5}$, a subnormal stored as the nearest multiple of $2^{-24}$: $1.024 \times 10^{-5} / 5.96 \times 10^{-8} \approx 171.8$, rounded to 172, giving $1.0252 \times 10^{-5}$.

Dividing by 1024 in fp32 recovers $1.0012 \times 10^{-8}$, within 0.12 percent of the truth.

</div>

## Step 4: fp8 needs a scale on every tensor

E4M3 spans only $2^{-6}$ to 448 in normal numbers, so raw tensors rarely fit. Every fp8 tensor carries a **scale factor** $s$; it stores $x/s$, with $s$ chosen so the largest entry lands at the format's maximum:

$$s = \frac{\max_i |x_i|}{448}, \qquad x \approx s \cdot \text{fp8}\!\left(\frac{x}{s}\right).$$

The common recipe uses E4M3 for forward weights and activations and E5M2 for gradients. NVIDIA's Transformer Engine uses **delayed scaling**: the scale comes from a history of recent maxima, so no extra pass over the data is needed.

<div class="ex">
<div class="lab">Worked example</div>

**quantising three numbers to E4M3**

Take $x = (0.1, 3, 300)$. The scale is $s = 300 / 448 = 0.6696$, so $x/s = (0.1493, 4.48, 448)$.

E4M3 has 3 mantissa bits, so between 4 and 8 the spacing is $4/8 = 0.5$: 4.48 rounds to 4.5. Between $0.125$ and $0.25$ the spacing is $0.125/8 = 0.0156$: $0.1493/0.0156 = 9.56$, rounded to 10, giving 0.15625. And 448 is exact.

Multiply back by $s$: $(0.1046, 3.013, 300)$, errors of 4.6, 0.45 and 0 percent, the normal price of 3 mantissa bits.

</div>

One scale per tensor is fragile. If one channel is thousands of times larger than the rest (LLM activations do develop such outlier channels), it sets $s$ for everyone and pushes ordinary values into the subnormal zone or to zero.

~~~dg-fig 13.04-fp8scale.svg
A 256 × 1024 tensor of ordinary values plus one outlier channel of growing size (synthetic). With one scale per tensor, the error on the ordinary values climbs from 2.3 percent once the outlier is about 1,000 times larger; with one scale per block of 128 values only the outlier's own blocks suffer.
~~~

Under per-tensor scaling the error on ordinary values is 2.3, 3.6 and 15 percent for no outlier, a 1,000-times and a 10,000-times outlier; per-block scales give 2.2, 2.3 and 2.6 percent. So recent large runs use **block-wise scaling** (DeepSeek-V3 is a published example: 1 × 128 tiles for activations, 128 × 128 blocks for weights). Even then fp8 covers only the big matmuls; embeddings, output head, norms, softmax and the optimiser stay in bf16 or fp32.

## Step 5: activation checkpointing

Precision shrinks tensors by a constant factor; activations need a bigger lever. From lesson 13.01, one transformer layer with 16-bit activations stores about

$$sbh\left(34 + \frac{5as}{h}\right) \text{ bytes},$$

where $s$ is sequence length, $b$ micro-batch size, $h$ hidden size and $a$ the number of heads. The $5as^2b$ part is the attention scores, softmax output and its dropout mask.

These tensors exist only because backward needs them. **Activation checkpointing** (also called rematerialisation) keeps just the input of each layer, $2sbh$ bytes, throws away the rest after forward, and just before that layer's backward runs its forward again to regenerate them. Peak activation memory becomes $L \times 2sbh$ plus one layer's full set.

The price is one extra forward pass. Forward is about $2N$ of the $6N$ FLOPs per token, so full recomputation raises the cost to about $8N$:

$$\frac{8N}{6N} = 1.33,$$

33 percent more compute. **Model FLOPs utilisation** (MFU) ignores recomputed work, so it drops even though the hardware is just as busy; **hardware FLOPs utilisation** (HFU) counts it.

~~~dg-fig 13.04-ckpt.svg
Left: activation memory through one training step of the 6.7B model from lesson 13.01 (analytic, fused attention). Storing everything peaks at 18.3 GB; checkpointing every layer keeps the curve nearly flat at 1.6 GB; every 4 layers sits in between. Right: three strategies with their FLOPs per token.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the 6.7B model's activation budget**

Shape from lesson 13.01: $L = 32$, $h = 4096$, $s = 4096$, micro-batch $b = 1$, $a = 32$ heads. So $sbh = 4096 \times 4096 = 16.8$ million.

Unfused attention: $5as/h = 5 \times 32 \times 4096 / 4096 = 160$, so each layer stores $(34 + 160) \times 16.8\text{M} = 3.25$ GB, and 32 layers need $104$ GB.

Selective recomputation or FlashAttention removes the 160 term: $34 \times 16.8\text{M} = 0.57$ GB per layer, $18.3$ GB in total, for a few percent of extra compute.

Full checkpointing keeps only layer inputs: $2 \times 16.8\text{M} = 33.6$ MB per layer, $1.07$ GB for 32 layers, plus one layer being recomputed ($0.57$ GB), about $1.6$ GB in total, for 33 percent more FLOPs.

</div>

## Step 6: recompute only what is cheap

Activations differ in cost. The quadratic $5as^2b$ term is huge but takes few FLOPs to rebuild; matmul outputs are modest to store but expensive to recompute. **Selective recomputation** (Korthikanti et al.) keeps the matmul outputs and recomputes only the attention core, which removes the quadratic term for a few percent of extra compute.

FlashAttention does this by construction: it never stores the score matrix and recomputes it tile by tile in backward. With sequence parallelism (13.03) dividing the remaining $34sbh$ by the tensor-parallel degree $t$, this is usually enough; full checkpointing is for when memory is still short.

~~~dg-flow Which memory lever to pull first, cheapest first.
Activations too big -> *FlashAttention or selective recompute -> Sequence parallelism with TP -> Checkpoint some layers -> Checkpoint every layer
~~~

## Mixed precision and checkpointing in code

Every printed value below is from an actual run (PyTorch 2.15, CPU).

```python
import torch
one = torch.tensor(1.0, dtype=torch.bfloat16)
print(one + 1e-4 == one)                         # tensor(True): the update vanished
print(torch.tensor(1e-8).half().item(),          # 0.0 (underflow)
      (torch.tensor(1e-8) * 1024).half().item()) # 1.0251998901367188e-05 (rescued)

x = torch.tensor([0.1, 3.0, 300.0])
s = x.abs().max() / 448                          # per-tensor fp8 scale
q = (x / s).to(torch.float8_e4m3fn)
print(q.float() * s)                             # tensor([1.0463e-01, 3.0134e+00, 3.0000e+02])
```

A standard bf16 training step. The parameters stay fp32 (they are the master weights); `autocast` casts them to bf16 inside matmuls, and the gradients land back in fp32.

```python
import torch, torch.nn as nn
model = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 10))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
X, y = torch.randn(512, 64), torch.randint(0, 10, (512,))
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):   # "cuda" on a GPU
    logits = model(X)
    loss = nn.functional.cross_entropy(logits.float(), y)        # loss in fp32
opt.zero_grad(); loss.backward(); opt.step()
print(logits.dtype, model[0].weight.dtype, model[0].weight.grad.dtype)
# torch.bfloat16 torch.float32 torch.float32
```

For fp16 on a GPU, add `scaler = torch.amp.GradScaler("cuda")` and use `scaler.scale(loss).backward()`, `scaler.step(opt)`, `scaler.update()`; bf16 needs none of that.

Checkpointing, with its saving measured. A saved-tensor hook adds up every tensor autograd keeps for backward, for 8 residual MLP blocks with input $16 \times 128 \times 256$ in fp32 (2 MiB each):

```python
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
class Block(nn.Module):
    def __init__(s, d): super().__init__(); s.ln = nn.LayerNorm(d); s.up = nn.Linear(d, 4*d); s.down = nn.Linear(4*d, d)
    def forward(s, x): return x + s.down(nn.functional.gelu(s.up(s.ln(x))))
blocks = nn.ModuleList(Block(256) for _ in range(8))
x = torch.randn(16, 128, 256, requires_grad=True)

def saved_mib(use_ckpt):
    total = 0
    def pack(t):
        nonlocal total; total += t.numel() * t.element_size(); return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        h = x
        for b in blocks:
            h = checkpoint(b, h, use_reentrant=False) if use_ckpt else b(h)
    h.sum().backward()
    return total / 2**20
print(saved_mib(False), saved_mib(True))   # 176.140625 16.0
```

Without checkpointing each block keeps 22 MiB (eleven input-sized tensors, mostly the 4x-wide MLP intermediates); with it only the 2 MiB input survives, 11 times less.

## Checking that it worked

- **Compare against a bf16 reference** run long enough: fp8 problems often appear only after many thousands of steps.
- **Watch the loss scale** in fp16 runs: one that keeps halving signals a real instability.
- **Track the update-to-weight ratio** $|\Delta w| / |w|$ per layer. Below about $2^{-9}$ to $2^{-8}$ (half the bf16 gap) with bf16 storage, updates are being rounded away.
- **Compare gradients with and without checkpointing** on a small fp32 model: recomputation should change memory and time, not numbers.
- **Report MFU and HFU together** when recomputation is on.

## What goes wrong

| symptom | likely cause | fix
| --- | --- | --- |
| loss flattens late in the schedule, no error | bf16 weights with no fp32 master copy; small updates rounded away | fp32 master weights, or stochastic rounding / Kahan summation
| sudden NaN or inf in fp16 | overflow above 65504, often in attention logits or a big activation | dynamic loss scaling; compute softmax in fp32; move to bf16
| fp16 run learns slowly, many zero gradients | gradient underflow, no or too little loss scaling | `GradScaler`, larger initial scale
| fp8 run tracks bf16, then drifts apart after many steps | per-tensor scales spoiled by outlier channels, stale delayed scales | block-wise scaling; keep sensitive layers in bf16
| out of memory even with checkpointing | the recomputed layer still holds the quadratic attention term | FlashAttention inside the layer
| MFU dropped about 25 percent after enabling checkpointing | recomputed forward is not counted as useful work ($6N/8N = 0.75$) | expected; try selective recompute first

## Going deeper

**Why rounding is relative.** With a fixed number of significant bits, relative error is at most $2^{-(m+1)}$ for normal numbers (the unit roundoff). So the update problem depends on $|u|/|w|$, and large weights are the hardest to update.

**Accumulation inside a matmul.** Random rounding errors in a length-$K$ dot product grow roughly like $\sqrt{K}$, which is why the sum must be kept in higher precision. DeepSeek-V3 reported that fp8 tensor-core accumulation was not precise enough at large $K$ and periodically promoted partial sums to fp32: "accumulate high" inside one matmul.

**The square-root schedule.** Chen et al. considered a chain of $n$ equal operations. Checkpointing every $k$-th activation stores $n/k$ checkpoints and, during backward, rebuilds one segment of $k$ activations at a time, so peak memory is about $n/k + k$, minimised at $k = \sqrt{n}$ with $O(\sqrt{n})$ memory for one extra forward. A transformer layer is not one operation: its $34sbh$ bytes are about 17 tensors of the $2sbh$ checkpoint size, so 32 layers are roughly $n = 544$ units and the optimum segment is about $\sqrt{544} \approx 23$ units, little more than one layer. So checkpointing every layer is near optimal for transformers, which is why "every 4 layers" peaks higher (2.6 GB) than "every layer" (1.6 GB) in the figure.

**Interactions.** Under pipeline parallelism (13.03) the first stage holds activations for the most in-flight micro-batches, so it gains most from checkpointing. Recomputation also reruns any tensor-parallel communication inside the layer, which frameworks try to overlap.

## Key terms

| Term | Meaning
| --- | --- |
| Exponent / mantissa bits | Bits that set a format's range / its precision
| bf16 | 16-bit format with fp32's range and 8 significant bits; no loss scaling needed
| Master weights | fp32 copy of the weights that the optimiser updates; cast down for compute
| Loss scaling | Multiplying the loss by $S$ so fp16 gradients do not underflow, undone before the update
| Subnormal | A number below the smallest normal value, stored with fewer significant bits
| fp8 scale factor | Per-tensor or per-block multiplier that maps a tensor into fp8's narrow window
| Activation checkpointing | Keeping only layer inputs in forward and recomputing the rest during backward
| Selective recomputation | Recomputing only cheap, memory-heavy activations such as attention scores
| MFU vs HFU | Utilisation counting only model FLOPs vs also counting recomputed FLOPs

<div class="take">
<div class="lab">Key takeaway</div>

compute in bf16 or fp8 with scaling, accumulate and update in fp32, and buy activation memory with selective recomputation first and full checkpointing only when still needed.

</div>
