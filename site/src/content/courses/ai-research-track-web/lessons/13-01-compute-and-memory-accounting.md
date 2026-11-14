---
id: "13.01"
course: "ai-research-track-web"
module: "13"
slug: "compute-and-memory-accounting"
title: "Compute and memory accounting"
keyIdeas: ["A dense decoder has about $12Ld^2$ parameters in its blocks ($4d^2$ for the attention projections, $8d^2$ for an MLP with a $4d$ hidden layer) plus $Vd$ per embedding table. At billions of parameters the blocks hold almost everything; in tiny models the embeddings can dominate.", "Training costs about $6N$ FLOPs per token: $2N$ forward (one multiply-add per weight) and $4N$ backward (an input gradient and a weight gradient per matrix). Attention's score and value products add about $12LTd$ per token, which is $T/(6d)$ of $6N$.", "Mixed-precision training with Adam keeps about 16 bytes per parameter before activations: bf16 weights and gradients (2 + 2) plus fp32 master weights and two fp32 moments (4 + 4 + 4). A 7B model's state does not fit on one 80 GB GPU.", "Activation memory grows with micro-batch size, sequence length, hidden size and depth, about $sbh(34 + 5as/h)$ bytes per layer. Without a fused attention kernel the stored score matrices make it quadratic in sequence length, so at long context activations run out before weights do.", "Convert FLOPs to time with achieved throughput, not peak: GPU-seconds = total FLOPs / (peak FLOP/s x MFU). Model FLOPs utilisation of 35 to 50 percent is typical for well-tuned dense training; measure it from tokens per second as soon as the run starts."]
question: "You are asked to cost a run: a dense decoder with 48 layers, hidden size 6144, vocabulary 50,000 (untied embeddings), context 2048, trained on 600B tokens on 256 H100s (989 TFLOP/s bf16 peak) at 45 percent MFU. Estimate the parameter count, the training FLOPs with and without the attention term, the wall-clock time, and the model-state memory. Can any single GPU hold the full optimiser state?"
answer: "Parameters: blocks $12 \\times 48 \\times 6144^2 = 21.7$B, embeddings $2 \\times 50{,}000 \\times 6144 = 0.61$B, so $N \\approx 22.4$B. Training FLOPs: $6 \\times 22.4\\text{B} \\times 600\\text{B} = 8.05 \\times 10^{22}$. The attention term is $12 \\times 48 \\times 2048 \\times 6144 \\times 600\\text{B} = 4.35 \\times 10^{21}$, about 5.4 percent extra, consistent with $T/(6d) = 2048/36{,}864 = 5.6$ percent (slightly lower because $N$ also contains the embeddings); short contexts make attention cheap relative to the weights. Total about $8.48 \\times 10^{22}$ FLOPs. Throughput: $256 \\times 989 \\times 10^{12} \\times 0.45 = 1.14 \\times 10^{17}$ FLOP/s, so about $7.4 \\times 10^5$ seconds, roughly 8.6 days, before restarts, evaluation and checkpointing overhead. Model state at 16 bytes per parameter: $16 \\times 22.4\\text{B} = 358$ GB. No single 80 GB GPU can hold it; even before activations the state must be split at least five ways, and in practice it is sharded across all 256 GPUs (about 1.4 GB each under full ZeRO-3 sharding) or split by tensor and pipeline parallelism. Activations come on top: at micro-batch 1 with FlashAttention, $2048 \\times 6144 \\times 34 \\approx 0.43$ GB per layer, about 21 GB over 48 layers, so the activation plan (micro-batch size, checkpointing) matters as well. The first planning decision is therefore how to partition state, not how big a batch to use."
sources: [{"title": "Kaplan et al.: Scaling Laws for Neural Language Models (2020)", "url": "https://arxiv.org/abs/2001.08361", "note": "Section 2.1 and Table 1: the per-operation parameter and FLOP counts behind the 6N rule."}, {"title": "Korthikanti et al.: Reducing Activation Recomputation in Large Transformer Models (2022)", "url": "https://arxiv.org/abs/2205.05198", "note": "Section 4: the per-layer activation memory derivation, term by term."}, {"title": "How to Scale Your Model (JAX scaling book)", "url": "https://jax-ml.github.io/scaling-book/", "note": "The chapter on Transformer maths: parameter, FLOP and memory counting with worked numbers."}, {"title": "Rajbhandari et al.: ZeRO (2019)", "url": "https://arxiv.org/abs/1910.02054", "note": "Section 3.1: where the 16 bytes per parameter of mixed-precision Adam come from."}]
practice: ["transformer-training-flop-estimator", "mixed-precision-training-memory-accountant", "training-compute-budget", "gradient-accumulation-equivalence"]
tags: ["FLOPs", "memory", "MFU"]
tools: ["PyTorch", "Python"]
---

## What you will learn

- How to count the parameters of a GPT-style decoder from its shape, and which parts of the model hold them.
- Why training costs about $6N$ floating-point operations per token, where that 6 comes from, and when attention adds a noticeable extra.
- Why mixed-precision training with Adam needs about 16 bytes per parameter before a single activation is stored.
- How activation memory grows with batch, sequence length, width and depth, and why naive attention makes it quadratic in sequence length.
- How to turn FLOPs into GPU-days with model FLOPs utilisation (MFU), and check every rule on a tiny model.

*Before you start:* you should know the transformer decoder block (attention plus MLP), backpropagation, the Adam optimiser and what bf16 and fp32 are. The PyTorch course covered `autocast` and optimisers; this lesson only needs the idea that a matrix multiply is many multiply-adds.

## Where this module starts

This module is about runs that need hundreds or thousands of GPUs, where a mistake in the plan costs weeks and a large bill. So it opens with arithmetic. A large run is a negotiation between **compute** (FLOPs: how long it takes), **memory** (bytes: whether it fits, and how it must be split) and **bandwidth** (whether GPUs compute or wait on each other).

The first two can be estimated on paper to within ten or twenty percent. Lessons 13.02 and 13.03 answer "the memory does not fit", 13.04 trades compute for memory, and 13.05 (scaling laws) splits a fixed FLOP budget between parameters and tokens. All of them start from the numbers in this lesson.

## The idea in one picture

Take a concrete plan: a dense decoder with $L = 32$ layers, hidden size $d = 4096$, 32 attention heads, a vocabulary of $V = 32{,}000$ tokens with separate input and output embeddings, context 4096, trained on one trillion tokens. Four lines of arithmetic give the whole budget:

| Quantity | Rule | Value
| --- | --- | --- |
| Parameters $N$ | $12Ld^2 + 2Vd$ | 6.70 billion
| Training compute $C$ | $6ND$ | $4.02 \times 10^{22}$ FLOPs
| Model state memory | 16 bytes per parameter | 107 GB
| Wall-clock time | $C / (\text{GPUs} \times \text{peak} \times \text{MFU})$ | 2.3 days on 512 H100s at 40% MFU

~~~dg-fig 13.01-memory.svg
The memory side of the plan. The 107 GB of optimiser state alone is larger than one 80 GB GPU, and so are the activations if attention scores are stored. Both facts force the sharding of lessons 13.02 and 13.03.
~~~

The top bar is the **model state**: weights, gradients and optimiser moments that live for the whole run. At 107 GB, this model cannot train on one 80 GB GPU however small the batch. The lower bars are **activations**, intermediate results the forward pass keeps for the backward pass.

A **FLOP** is one floating-point multiply or add ("FLOP/s" is a rate). **MFU** (model FLOPs utilisation) is the fraction of the hardware's peak FLOP/s spent on the maths the model requires. The steps below derive each line of the table.

## Step 1: count the parameters

Almost all parameters sit in attention and MLP weight matrices, each of size proportional to $d^2$. Per layer of a GPT-style block with hidden size $d$:

- The attention projections $W_Q, W_K, W_V, W_O$ are each $d \times d$: $4d^2$ parameters.
- The MLP has an up-projection $d \times 4d$ and a down-projection $4d \times d$: $8d^2$ parameters.
- LayerNorm scales and biases are vectors of length $d$: negligible.

So the blocks hold $12Ld^2$ parameters, where $L$ is the number of layers. Add the embedding tables: the input embedding is $V \times d$ (one vector per vocabulary entry) and, if it is not shared ("tied") with the input, the output projection to the vocabulary is another $V \times d$.

$$N \approx 12Ld^2 + 2Vd \quad \text{(untied embeddings)}$$

| Component | Parameters per layer | Note
| --- | --- | --- |
| attention Q, K, V, O | $4d^2$ | grouped-query attention shrinks K and V
| MLP with $4d$ hidden | $8d^2$ | SwiGLU with hidden $8d/3$ and three matrices is also $8d^2$
| norms, biases | $O(d)$ | ignore for budgeting
| embeddings | $Vd$ per table | doubled if input and output are untied

Number check with the plan above: $12 \times 32 \times 4096^2 = 6.44$ billion in the blocks, and $2 \times 32{,}000 \times 4096 = 0.26$ billion in the embeddings, so $N = 6.70$ billion.

~~~dg-fig 13.01-params.svg
Where the parameters live. In a tiny model the embedding tables are most of the model; from a few billion parameters up the split settles at about one third attention and two thirds MLP, with embeddings a few percent.
~~~

So "a 20M-parameter model" may have only 3M parameters doing sequence processing, while at scale $12Ld^2$ alone is accurate to a few percent. Mixture-of-experts (MoE) models need two counts: **total parameters** (memory) and **active parameters** per token (compute); with 2 of 8 experts used per token, all 8 MLPs are stored but only 2 compute.

## Step 2: FLOPs per token, the 6N rule

Start from one matrix multiply. Multiplying a $1 \times k$ activation row by a $k \times n$ weight matrix produces $n$ outputs, each a dot product of length $k$: $k$ multiplies and about $k$ adds. That is

$$\text{FLOPs} = 2kn = 2 \times (\text{number of weights}).$$

Summed over every weight matrix, one token's **forward pass** costs about $2N$ FLOPs.

The **backward pass** does two matrix multiplies for each weight matrix. For a layer $Y = XW$ it needs the gradient with respect to the input, $\partial X = \partial Y\, W^\top$ (so the gradient can flow to earlier layers), and the gradient with respect to the weight, $\partial W = X^\top \partial Y$ (so the weight can be updated). Each costs as much as the forward multiply, so backward costs about $4N$. Training on $D$ tokens therefore costs

$$C \approx 6ND,$$

where $C$ is total training FLOPs, $N$ the parameter count and $D$ the number of training tokens. Number check: $6 \times 6.70 \times 10^9 \times 10^{12} = 4.02 \times 10^{22}$.

Two refinements matter in practice:

- **The input embedding costs nothing**: a table lookup is a memory read, not a matmul (the output projection does count). Leave the input table out of $N$ when checking against a measurement.
- **Attention has its own matmuls.** $QK^\top$ and the product with $V$ have no weights, so $6N$ misses them. Each query dots with $T$ keys of length $d$ (about $2Td$ FLOPs per layer) and mixes $T$ values (another $2Td$): $4LTd$ forward per token, about $12LTd$ for training.

Relative to $6N \approx 72Ld^2$, the attention extra is

$$\frac{12LTd}{72Ld^2} = \frac{T}{6d}.$$

At $T = d = 4096$ that is about 17 percent. It grows linearly with context length and shrinks with width.

~~~dg-fig 13.01-flops.svg
Left: on five tiny GPTs, PyTorch's FLOP counter agrees with 6N plus 12LTd to within 0.3 percent; the measured count is 3 to 6 percent above 6N alone at short context and 28 percent above it at T = 512 with d = 256. Right: the attention share T/(6d) as a function of context length for three widths.
~~~

The left panel is a measurement: tiny GPTs run through one forward and backward pass under `torch.utils.flop_counter.FlopCounterMode`, divided by the token count. Causal masking makes half the score matrix unnecessary and fused kernels skip it, so the true attention cost is nearer $6LTd$. Reports differ on whether they count attention at all, so check the convention before comparing MFU figures.

## Step 3: memory for the model state

~~~dg-layers Bytes per parameter in mixed-precision Adam training (before activations).
bf16 weights | 2 bytes, used in forward and backward
bf16 gradients | 2 bytes, produced by backward
*fp32 master weights | 4 bytes, what the optimiser actually updates
fp32 Adam first moment | 4 bytes, running mean of gradients
fp32 Adam second moment | 4 bytes, running mean of squared gradients
~~~

That totals 16 bytes per parameter. Number check: $16 \times 6.70 \times 10^9 = 107$ GB, the top bar of the first figure.

Why an fp32 master copy? Updates are often $10^{-3}$ to $10^{-5}$ times the weight, and bf16 keeps about 8 bits of precision, so such a change added in bf16 often rounds away. The fp32 copy accumulates small steps; the bf16 copy is re-cast from it.

A surprise: plain fp32 Adam is *also* 16 bytes per parameter (4 weight + 4 gradient + 8 for the moments; stepping `torch.optim.AdamW` on a 4.2M-parameter model confirms exactly that). Mixed precision does not shrink the state; it shrinks activations and doubles matmul throughput. Some frameworks keep gradients in fp32 (18 to 20 bytes); 8-bit optimisers or bf16 moments go lower, at some risk to stability.

## Step 4: activation memory

Backward needs the inputs of every matmul and non-linearity from forward. These saved **activations** scale with how much data is in flight.

For a GPT-style layer with 16-bit activations, Korthikanti et al. estimate

$$\text{bytes per layer} \approx s\,b\,h\left(34 + \frac{5as}{h}\right),$$

where $s$ is sequence length, $b$ the micro-batch size (sequences per GPU per step), $h$ the hidden size (the same as $d$) and $a$ the number of attention heads.

- $34sbh$ is linear in $s$: layer inputs, Q, K, V, the MLP's $4h$-wide intermediates, norm inputs, dropout masks.
- $5as^2b$ is the scores, softmax output and its dropout mask: $s \times s$ per head, quadratic in $s$.

FlashAttention-style kernels work in tiles and never store the $s \times s$ matrices, so the figure drops to about $34sbh$.

Number check for the plan, micro-batch 1: $sbh = 4096 \times 1 \times 4096 = 16.8$ million. Without a fused kernel, $5as/h = 5 \times 32 \times 4096/4096 = 160$, so a layer needs $16.8\text{M} \times 194 = 3.25$ GB and 32 layers need 104 GB. With FlashAttention, $16.8\text{M} \times 34 = 0.57$ GB per layer and 18 GB in total.

~~~dg-fig 13.01-activations.svg
Left: measured on CPU, the tensors one block saves for backward grow linearly with sequence length with PyTorch's fused attention, but bend upward when the score matrix is stored. Right: the same split predicted by the formula for one layer of the 6.7B model.
~~~

The left panel uses `torch.autograd.graph.saved_tensors_hooks` on one fp32 block ($d = 256$, 8 heads). With `scaled_dot_product_attention` it saved exactly 16 numbers per token per hidden unit at every length (8 of them are the MLP's two $4d$-wide tensors); in bf16 that is 32 bytes per $sbh$, close to the formula's 34 (the test block has no dropout). Hand-written attention also saved one score matrix per head: 172 MB against 34 MB at $s = 2048$.

Do not forget the **logits**: $b \times s \times V$ values, often in fp32 for the loss. At $s = 4096$ and $V = 128{,}000$ that is about 2.1 GB per sequence, hence fused or chunked cross-entropy kernels.

## Step 5: from FLOPs to wall-clock time

A GPU's **peak** rate for an H100 SXM is about 989 TFLOP/s for dense bf16 matmuls (spec sheet value). Memory-bound operations, communication, pipeline bubbles and data loading keep real training well below it. MFU measures how much you keep:

$$\text{MFU} = \frac{\text{model FLOPs per token} \times \text{tokens per second}}{n_{\text{GPU}} \times \text{peak FLOP/s}}.$$

Recomputed activations (lesson 13.04) are extra hardware work that MFU does not credit; **hardware FLOPs utilisation** (HFU) does, so HFU is always at least MFU.

Rearranged, the schedule is

$$t = \frac{C}{n_{\text{GPU}} \times \text{peak} \times \text{MFU}}.$$

Well-tuned dense training typically reaches roughly 35 to 50 percent MFU; small models, long sequences or heavy communication do worse.

~~~dg-fig 13.01-time.svg
Wall-clock time for the 6.7B, one-trillion-token run. Time falls as 1/MFU, so moving from 30 to 45 percent MFU saves as much as adding half again as many GPUs.
~~~

~~~dg-flow From model shape to a training schedule.
Shape (L, d, V) -> Parameters N -(x 6 x D)-> *Training FLOPs C -(peak x MFU x GPUs)-> Wall-clock days
Parameters N -(x 16 bytes)-> State memory -> Sharding plan
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a 6.7B model on one trillion tokens**

Shape: $L = 32$, $d = 4096$, 32 heads, $V = 32{,}000$, untied embeddings, context 4096. Parameters: $12 \times 32 \times 4096^2 = 6.44$B in blocks plus $2 \times 32{,}000 \times 4096 = 0.26$B in embeddings, so $N = 6.70$B.

Compute: $6 \times 6.70\text{B} \times 10^{12} = 4.02 \times 10^{22}$ FLOPs. Attention adds $12 \times 32 \times 4096 \times 4096 = 6.4$ GFLOPs per token against $6N = 40.2$ GFLOPs per token, 16 percent more.

Time at 40 percent MFU on H100s: one GPU delivers $989 \times 10^{12} \times 0.4 = 3.96 \times 10^{14}$ FLOP/s. Then $4.02 \times 10^{22} / 3.96 \times 10^{14} = 1.02 \times 10^8$ GPU-seconds, about 28,200 GPU-hours. Divided over 512 GPUs: 55 hours, or 2.3 days. Including the attention term it is 2.7 days.

Memory: $16 \times 6.70\text{B} = 107$ GB of state. Activations at micro-batch 1 without a fused attention kernel: $4096 \times 4096 \times (34 + 160) = 3.25$ GB per layer, 104 GB across 32 layers. With FlashAttention: 0.57 GB per layer, 18 GB in total.

Verdict: both the state and the naive activations exceed one 80 GB GPU. The state must be sharded (13.02) and the attention kernel must be fused; micro-batch size is decided last.

</div>

## The budget in code

A planner is ten lines of Python; keep one next to every run config.

```python
def plan(L, d, V, T, D, a, gpus, peak=989e12, mfu=0.40, tied=False, flash=True, mb=1):
    N = 12*L*d*d + (1 if tied else 2)*V*d          # parameters
    C = 6*N*D                                       # training FLOPs, weights only
    C_att = 12*L*T*d*D                              # attention scores and values
    days = (C + C_att) / (gpus*peak*mfu) / 86400
    state_gb = 16*N/1e9                             # bf16 w+g, fp32 master + 2 moments
    per_layer = T*mb*d*(34 + (0 if flash else 5*a*T/d))
    act_gb = L*per_layer/1e9
    return dict(N_B=N/1e9, C=C, att_pct=100*C_att/C, days=days,
                state_GB=state_gb, act_GB=act_gb)

print(plan(L=32, d=4096, V=32_000, T=4096, D=1e12, a=32, gpus=512))
# N_B 6.705  C 4.023e+22  att_pct 16.02  days 2.667  state_GB 107.3  act_GB 18.25
```

The formulas are only worth trusting if you have checked them against a real model once. PyTorch can count FLOPs and bytes for you on a tiny GPT that runs on a CPU in seconds:

```python
import torch, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
# GPT(L, d, V): a ~25-line nn.Module: embedding, L pre-norm blocks
# (fused qkv, out-proj, 4d GELU MLP, no biases), final norm, untied head
m = GPT(L=4, d=256, V=2000); T = 64
idx = torch.randint(0, 2000, (1, T))
N_compute = sum(p.numel() for p in m.parameters()) - 2000*256   # drop the lookup table
with FlopCounterMode(display=False) as fc:
    F.cross_entropy(m(idx).view(-1, 2000), idx.view(-1)).backward()
print(fc.get_total_flops()/T, 6*N_compute, 6*N_compute + 12*4*T*256)
# 22732800.0  21974016  22760448     -> measured is 6N + 12LTd within 0.2%

opt = torch.optim.AdamW(m.parameters()); opt.step()
state = sum(t.numel()*t.element_size() for s in opt.state.values()
            for t in s.values() if t.dim() > 0)
print(state / sum(p.numel() for p in m.parameters()))
# 8.0   -> two fp32 moments; plus 4 (weight) + 4 (grad) = 16 bytes/param
```

## Checking the plan against the run

Once the job runs, three numbers test the plan.

**1. Measured MFU.** Log tokens per second and compute MFU with the formula from step 5. Suppose 512 GPUs process 4.0 million tokens per second on the 6.7B model (a hypothetical reading). Then MFU $= 6 \times 6.70\text{B} \times 4.0\text{M} / (512 \times 989 \times 10^{12}) = 1.61 \times 10^{17} / 5.06 \times 10^{17} \approx 32$ percent. Against a planned 40 percent the run takes 25 percent longer; look for communication stalls or a slow data loader.

**2. Peak memory per GPU.** Compare `torch.cuda.max_memory_allocated()` after a few steps with predicted state plus activations. A gap usually means fp32 logits, an unfused attention path or communication buffers.

**3. Parameter count.** `sum(p.numel() for p in model.parameters())` should match $12Ld^2 + 2Vd$ within a few percent. If not, check the MLP ratio and embedding tying.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Using peak FLOP/s as achieved | schedule is 2 to 3 times too optimistic | plan with 35 to 50 percent MFU, then measure
| Counting only weights and gradients | out-of-memory at step 1 of the optimiser | budget 16 bytes per parameter for Adam
| Forgetting activations | fits at short context, crashes at long context | compute $sbh(34 + 5as/h)$ per layer, use fused attention
| Ignoring fp32 logits | memory spike at the loss, worst with large vocabularies | fused or chunked cross-entropy
| Comparing MFU across reports | "our 55 percent beats their 45" when conventions differ | check whether attention and causal masking were counted
| Counting recomputation as useful work | MFU above what the run's speed supports | report recomputation under HFU, not MFU
| Using total MoE parameters for compute | FLOP estimate several times too high | use active parameters for $C$, total for memory

## Going deeper

**Why backward is exactly twice forward.** For each weight matrix the forward pass does one matmul; the backward pass does two of the same shape (input gradient and weight gradient).

**Inference is $2N$ per token.** Generation is forward only: about 13.4 GFLOPs per token for the 6.7B model, plus attention over the KV cache. Decoding one token at a time is memory-bound, dominated by reading weights rather than by FLOPs.

**Grouped-query attention changes the count.** With $n_{kv}$ key-value heads of size $d_{head}$, $W_K$ and $W_V$ are $d \times n_{kv}d_{head}$ instead of $d \times d$. With 32 query heads and 8 KV heads, the attention block holds $2d^2 + 2 \times d^2/4 = 2.5d^2$ instead of $4d^2$.

**Where the constant 34 comes from.** Korthikanti et al. count, per layer and per token, in bf16 units of $h$: 11 for attention (the input, Q, K, V, the input to the output projection and a dropout mask), 19 for the MLP (the input, the $4h$ up-projection output, the $4h$ GELU output and a dropout mask) and 4 for the two LayerNorm inputs. SwiGLU, no dropout or a different MLP ratio shift the constant, so treat it as an estimate that tells you which term dominates rather than an exact byte count.

## Key terms

| Term | Meaning
| --- | --- |
| FLOP | One floating-point multiply or add; FLOP/s is the rate
| $6N$ rule | Training costs about 6 FLOPs per parameter per token: 2 forward, 4 backward
| Attention FLOPs | About $12LTd$ per token for training, or $T/(6d)$ of $6N$
| Model state | Weights, gradients and optimiser moments: about 16 bytes per parameter with mixed-precision Adam
| Master weights | The fp32 copy of the weights that the optimiser updates
| Activations | Forward-pass tensors saved for backward; scale with $s \cdot b \cdot h \cdot L$
| MFU | Required model FLOP/s divided by hardware peak FLOP/s
| HFU | Like MFU but also counts recomputed work
| Active parameters | In MoE, the parameters actually used for one token; they set compute

<div class="take">
<div class="lab">Key takeaway</div>

$N \approx 12Ld^2$, $C \approx 6ND$, 16 bytes per parameter of Adam state, and activations that scale with $s \cdot b \cdot h \cdot L$ (quadratically in $s$ without a fused attention kernel); a few lines of arithmetic, checked once against a tiny model, tell you whether a plan fits and how long it will take.

</div>
