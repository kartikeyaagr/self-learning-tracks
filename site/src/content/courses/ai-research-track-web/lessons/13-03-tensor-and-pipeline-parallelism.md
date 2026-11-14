---
id: "13.03"
course: "ai-research-track-web"
module: "13"
slug: "tensor-and-pipeline-parallelism"
title: "Tensor and pipeline parallelism"
keyIdeas: ["Tensor parallelism splits each weight matrix across GPUs. In Megatron's MLP the first matrix is split by columns (so GeLU runs locally) and the second by rows, so the only communication is one all-reduce of the layer output in forward and one in backward.", "Each Transformer layer under tensor parallelism needs two all-reduces of a $b \\times s \\times h$ activation in forward and two in backward, on the critical path. That is why tensor parallelism stays inside a node on NVLink.", "Pipeline parallelism assigns contiguous groups of layers to stages and streams micro-batches through them. With $p$ stages and $m$ micro-batches, idle time (the **bubble**) is $(p-1)/(m+p-1)$ of the step for both GPipe and 1F1B, so you need $m \\gg p$.", "1F1B keeps the same bubble as GPipe but caps stored activations at about $p$ micro-batches instead of $m$; interleaving $v$ model chunks per GPU shrinks the bubble by about $v$ at the price of more point-to-point messages.", "3-D parallelism nests the three by bandwidth need: tensor parallel within a node, pipeline across nodes, data parallel (often with ZeRO-1) outermost, with rank order chosen to match the network."]
question: "You have 512 GPUs (64 nodes of 8) for a 70B dense model, a global batch of 512 sequences of 4096 tokens, and micro-batch size 1. Compare layout A (TP 8, PP 8, DP 8) with layout B (TP 8, PP 4, DP 16): number of micro-batches per replica, bubble fraction with 1F1B, and model-state memory per GPU (16 bytes per parameter, before any ZeRO). Why would putting TP 16 across two nodes be a poor third option?"
answer: "Layout A: each of the 8 data-parallel replicas gets 512/8 = 64 sequences, so with micro-batch 1 there are $m = 64$ micro-batches and the bubble is $(8-1)/(64+7) = 7/71 \\approx 9.9$ percent. The model is split 64 ways (TP 8 x PP 8), so model state per GPU is $70\\text{B} \\times 16 / 64 = 17.5$ GB. Layout B: 16 replicas get 32 sequences each, $m = 32$, bubble $3/(32+3) = 3/35 \\approx 8.6$ percent. The model is split only 32 ways: $70 \\times 16 / 32 = 35$ GB per GPU, and each stage holds twice as many layers, so its activations (up to about $p$ micro-batches of its own layers under 1F1B) are larger too. B has a slightly smaller bubble, because fewer stages outweigh the halved $m$, and more data parallelism, but roughly twice the state; on 80 GB GPUs it fits only if activations stay well under about 40 GB, and ZeRO-1 across the 16 data-parallel ranks can recover most of the optimiser share (12 of the 16 bytes, so state drops to about $35 \\times (4 + 12/16)/16 \\approx 10.4$ GB). TP 16 across two nodes would put four all-reduces per layer of a $1 \\times 4096 \\times 8192$ bf16 activation (about 67 MB each) on the inter-node network, on the critical path of every layer. With roughly 50 GB/s per GPU instead of 450 GB/s on NVLink, each all-reduce takes milliseconds rather than a quarter of one, which outweighs the halved matmul time and any bubble savings. Keep tensor parallelism within the NVLink domain and use pipeline or data parallelism to cross nodes."
sources: [{"title": "Shoeybi et al.: Megatron-LM (2019)", "url": "https://arxiv.org/abs/1909.08053", "note": "Section 3: the column-then-row split of the MLP and attention, with the f and g operators."}, {"title": "Narayanan et al.: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (2021)", "url": "https://arxiv.org/abs/2104.04473", "note": "The interleaved 1F1B schedule and the guidance on combining tensor, pipeline and data parallelism."}, {"title": "Huang et al.: GPipe (2018)", "url": "https://arxiv.org/abs/1811.06965", "note": "Micro-batch pipelining and the bubble analysis that later schedules improve on."}, {"title": "How to Scale Your Model (JAX scaling book)", "url": "https://jax-ml.github.io/scaling-book/", "note": "The chapters on sharded matmuls and Transformer parallelism, with communication costs per strategy."}]
practice: ["reconstruct-column-parallel-mlp-layer", "measure-pipeline-bubble-utilization", "choose-topology-aware-parallel-plan", "tensor-parallel-mlp", "tensor-parallel-communication-cost"]
tags: ["tensor parallelism", "pipeline parallelism", "3-D parallelism"]
tools: ["Megatron-LM", "PyTorch", "NCCL"]
---

## What you will learn

- Why splitting the model's *computation*, not just its storage, becomes necessary.
- How tensor parallelism cuts one Transformer layer across GPUs (Megatron's column-then-row split) and where it communicates.
- How pipeline parallelism cuts the model into stages, why that creates idle time (the **bubble**), and how to compute it.
- Why 1F1B has GPipe's bubble but far less activation memory, and what interleaving buys.
- How to combine tensor, pipeline and data parallelism (3-D parallelism) on a real cluster.

*Before you start:* lesson 13.01 (counting parameters, FLOPs and bytes; there the hidden size is called $d$, here it is $h$ because $d$ will mean the data-parallel degree) and lesson 13.02 (data parallelism, the ring all-reduce and its cost $2\frac{n-1}{n}S$, ZeRO).

## When sharding is not enough

Lesson 13.02 ended with ZeRO-3: each GPU stores a slice of the model state and gathers each layer's weights just before use. That solves *storage*, but every GPU still computes every layer on its own micro-batch and keeps every layer's activations.

That breaks when one layer's matmuls or activations are too big for one GPU, when the global batch is already as large as optimisation tolerates, or when gathering whole layers every micro-batch costs more traffic than you can hide. The fix is to split the model's computation itself. There are two natural cuts. **Tensor parallelism** (TP) cuts *within* a layer: each weight matrix is split across GPUs and they cooperate on every matmul. **Pipeline parallelism** (PP) cuts *between* layers: each GPU owns a contiguous block of layers and passes activations to the next.

## The idea in one picture

Take one Transformer layer's MLP with hidden size $h = 8192$ and one sequence of $s = 4096$ tokens. It is two matrices: $A$ of shape $h \times 4h$ (up-projection) and $B$ of shape $4h \times h$ (down-projection). Together they hold $8h^2 \approx 537$ million parameters, about 1.07 GB in bf16, in a single layer.

Give it to 4 GPUs: cut $A$ into 4 vertical strips (column blocks) and $B$ into 4 horizontal strips (row blocks), and give GPU $i$ strip $i$ of each.

~~~dg-fig 13.03-split.svg
The Megatron MLP on 4 GPUs. GPU i holds only the blocks of colour i. The nonlinearity runs locally on each column block of Y; the only communication is one sum (all-reduce) of the four partial outputs.
~~~

| What GPU $i$ holds or computes | Shape with $t = 4$ | Communication
| --- | --- | --- |
| input $X$ | $4096 \times 8192$, same on all GPUs | none
| strip $A_i$ of $A$ | $8192 \times 8192$ (a quarter of the columns) | none
| $Y_i = \mathrm{GeLU}(XA_i)$ | $4096 \times 8192$ | none (GeLU is elementwise)
| strip $B_i$ of $B$ | $8192 \times 8192$ (a quarter of the rows) | none
| partial output $Y_iB_i$ | $4096 \times 8192$ | one all-reduce sums the 4 partials

Each GPU stores a quarter of the weights, does a quarter of the FLOPs, and the four GPUs talk exactly once in the forward pass, to add up their partial answers.

## Step 1: split the first matrix by columns

Write the MLP as $Z = \mathrm{GeLU}(XA)\,B$, where $X$ is the input activation (shape $s \times h$; with a batch of $b$ sequences it is $b \times s \times h$), $A$ and $B$ are the weights above, and $t$ is the number of tensor-parallel GPUs.

Splitting $A$ by columns, $A = [A_1, A_2, \dots, A_t]$, gives

$$XA = [XA_1, XA_2, \dots, XA_t].$$

Each column block of the product needs only the full $X$ and one column block of $A$, so GPU $i$ computes $Y_i = \mathrm{GeLU}(XA_i)$ alone. GeLU acts on each number separately, so applying it block by block changes nothing.

Why not split $A$ by rows? Then $XA = \sum_i X_iA_i$ is a *sum* of partial products, and the sum must happen before GeLU, because $\mathrm{GeLU}(a + b) \neq \mathrm{GeLU}(a) + \mathrm{GeLU}(b)$. That costs an extra all-reduce mid-MLP; the code below shows that summing after GeLU instead is simply wrong.

## Step 2: split the second matrix by rows

After step 1, GPU $i$ holds a column block $Y_i$ of $Y$. Matrix multiplication can consume column blocks directly if the next matrix is split by rows to match: $B = [B_1; B_2; \dots; B_t]$ stacked vertically. Then

$$Z = YB = \sum_{i=1}^{t} Y_i B_i.$$

GPU $i$ computes the partial product $Y_iB_i$, which has the full output shape $s \times h$, and one **all-reduce** (sum across GPUs, ending with the result on every GPU) produces $Z$. The intermediate $Y$, 4 times wider than the hidden size, never moves.

Megatron-LM writes this with two conjugate operators placed around the block:

- $f$ at the input: identity in forward (every GPU already has $X$), all-reduce in backward (each GPU computes only its share of $\partial L/\partial X$, and the shares must be summed).
- $g$ at the output: all-reduce in forward (sum the partials), identity in backward (the gradient $\partial L/\partial Z$ is the same on every GPU, so each just uses it).

~~~dg-flow Megatron tensor-parallel MLP as seen by one of the t GPUs.
X (replicated) -(f: identity forward, all-reduce backward)-> X A_i (column shard) -> GeLU (local) -> Y_i B_i (row shard) -(g: all-reduce forward)-> *Z (replicated)
~~~

## Step 3: attention, embeddings and the full layer

Attention splits the same way. Q, K and V are column-parallel, with whole **attention heads** as the blocks: with 64 heads and $t = 8$, each GPU owns 8 heads and runs their attention locally, since heads do not interact. The output projection $W_O$ is row-parallel and ends with an all-reduce, like $B$.

Per layer that is two all-reduces in forward (after attention and after the MLP) and two in backward (the $f$ operators): **four all-reduces of a $b \times s \times h$ tensor per layer per step**, all on the **critical path**: the next operation needs the result, so the GPU waits.

The embedding and output layers are split along the vocabulary; the softmax normaliser is combined with small all-reduces of per-token scalars, so full logits are never gathered.

**Sequence parallelism** (Korthikanti et al.) handles LayerNorm and dropout, which plain TP duplicates on every GPU. It splits those regions along the sequence and replaces each all-reduce by a **reduce-scatter** plus a later **all-gather**: the same total volume (a ring all-reduce is exactly those two phases, 13.02), but $t$ times less activation memory there.

## Step 4: why tensor parallelism lives inside a node

How big are those all-reduces? With $b = 1$, $s = 4096$, $h = 8192$ and bf16 (2 bytes):

$$S = 4096 \times 8192 \times 2 = 67.1 \text{ MB per all-reduce}.$$

<div class="ex">
<div class="lab">Worked example</div>

**is tensor-parallel communication affordable?**

Setup: $t = 8$, one 4096-token sequence, $h = 8192$. Spec-sheet values for an H100 SXM: 989 TFLOP/s dense bf16, NVLink 900 GB/s total (450 GB/s each way); one 400 Gb/s InfiniBand NIC (50 GB/s) per GPU. These are estimates, not measurements.

Bytes each GPU sends in one ring all-reduce: $2 \cdot \frac{t-1}{t} \cdot S = 2 \times \frac{7}{8} \times 67.1 = 117.4$ MB.

Time on NVLink: $117.4 \text{ MB} / 450 \text{ GB/s} = 0.26$ ms. Two all-reduces in the forward pass: 0.52 ms.

Compute for one layer forward: weight matmuls $24sh^2 = 6.60$ TFLOP plus attention $4s^2h = 0.55$ TFLOP, total 7.15 TFLOP, or 0.89 TFLOP per GPU. At 50 percent of peak: $0.89 / 494.5 = 1.81$ ms.

So inside a node, communication adds $0.52 / (1.81 + 0.52) = 22$ percent. Over InfiniBand the same all-reduce takes $117.4 / 50 = 2.35$ ms, two of them 4.7 ms, and communication becomes 72 percent of the layer time.

</div>

~~~dg-fig 13.03-comm.svg
Estimated forward time of one layer under tensor parallelism, built from spec-sheet bandwidth and FLOP numbers (not measured). On NVLink the two all-reduces are a modest slice; over the inter-node network they dominate, and going to 16 GPUs halves compute but makes the total worse.
~~~

The bottom bar shows why "TP 16 across two nodes" fails: compute halves (to 0.9 ms), but the ring is now limited by the inter-node link, and the layer takes 5.9 ms instead of 2.3 ms with TP 8 inside a node. This is why tensor parallelism is normally capped at the number of GPUs connected by NVLink, typically 8. Achieved NCCL bandwidth is below the spec values, so real communication shares are larger still.

## Step 5: pipeline parallelism and the bubble

Pipeline parallelism takes the other cut. With $L$ layers and $p$ **stages**, stage 1 gets layers $1$ to $L/p$, stage 2 the next block, and so on. Stages exchange only the activation at the boundary (forward) and its gradient (backward), with point-to-point sends. That is one $b \times s \times h$ message per micro-batch per boundary, instead of four all-reduces per *layer*, so pipelines tolerate the slower inter-node network well.

The catch is waiting: stage 2 cannot start until stage 1 is done. So the batch is cut into $m$ **micro-batches** and streamed; while stage 2 works on micro-batch 1, stage 1 starts micro-batch 2.

The simplest schedule, **GPipe**, runs all $m$ forwards, then all $m$ backwards. The top half of the next figure is an exact simulation with $p = 4$ stages and $m = 8$ micro-batches (a backward takes twice as long as a forward, which is typical). The white triangles are the **bubble**: time when a GPU is idle.

~~~dg-fig 13.03-schedule.svg
Simulated 4-stage pipeline with 8 micro-batches. Top: GPipe. Bottom: 1F1B. Numbers in boxes are micro-batch ids. Both schedules finish at the same time with the same 27.3 percent idle time; they differ in the order of work, which matters for memory.
~~~

Count the idle slots. Measured in "slots" of one forward plus one backward, each stage does $m$ slots of work. The last stage must wait $p - 1$ slots before its first micro-batch arrives, and the first stage waits $p - 1$ slots at the end while the backwards drain back to it. So each stage spends $m + p - 1$ slots in total, of which $p - 1$ are idle:

$$\text{bubble fraction} = \frac{p-1}{m+p-1}.$$

Here $p$ is the number of stages and $m$ the micro-batches per optimiser step. Check: $p = 4$, $m = 8$ gives $3/11 = 27.3$ percent, exactly what the simulation reports (33 time units, 24 of them work). Some papers divide by ideal compute time instead, giving $(p-1)/m$: same idle time, different denominator. Either way you need $m \gg p$.

~~~dg-fig 13.03-bubble.svg
Idle share of the step against the number of micro-batches, for 4, 8 and 16 stages. Dots are simulated 1F1B runs for p = 8 and land on the formula. The dashed line is the interleaved schedule with v = 2.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**bubbles for an 8-stage pipeline**

With $p = 8$, the idle fraction is $7/(m+7)$. For $m = 8$: $7/15 = 46.7$ percent idle. For $m = 16$: $7/23 = 30.4$ percent. For $m = 32$: $7/39 = 17.9$ percent. For $m = 64$: $7/71 = 9.9$ percent. The simulator in the figure gives the same four numbers.

Layout: 1,024 GPUs as TP 8 x PP 8 x DP 16. A global batch of 1,024 sequences gives each data-parallel replica $1024/16 = 64$ sequences. With micro-batch size 1 that is $m = 64$ and a 9.9 percent bubble.

Halving the global batch to 512 sequences halves $m$ to 32 and nearly doubles the bubble to 17.9 percent.

Interleaving with $v = 2$ (step 7) at $m = 32$ gives a bubble-to-compute ratio of $(p-1)/(vm) = 7/64 = 0.109$, which is $0.109/1.109 = 9.9$ percent of the step: the same as $m = 64$ without interleaving.

</div>

## Step 6: 1F1B, same bubble, bounded memory

GPipe has a memory problem. Each forward must keep its activations until the matching backward runs, and GPipe runs all $m$ forwards first, so stage 1 holds activations for all $m$ micro-batches at its peak. Fixing the bubble with a large $m$ makes memory worse.

The **1F1B** schedule ("one forward, one backward") reorders the same work. Stage $i$ runs a short warm-up of $p - i$ forwards, then alternates: one forward, one backward, starting each micro-batch's backward as soon as its gradient arrives. In the schedule figure the white area is unchanged, but backwards start much earlier.

~~~dg-fig 13.03-memory.svg
How many micro-batches' activations stage 1 must hold over time (p = 4, m = 8). GPipe climbs to all 8; 1F1B levels off at p = 4 and stays there no matter how large m becomes.
~~~

Under 1F1B, stage 1 holds at most $p$ micro-batches of activations (later stages hold fewer), independent of $m$. You can raise $m$ to shrink the bubble without paying in memory, which is why 1F1B is the usual default.

## Step 7: interleaving and newer schedules

The **interleaved** schedule (Narayanan et al., the Megatron 3-D paper) gives each GPU $v$ non-contiguous chunks of layers. With 32 layers, 8 GPUs and $v = 2$, GPU 1 holds layers 1-2 and 17-18, GPU 2 holds 3-4 and 19-20, and so on. Each visit does half the work, so warm-up and drain are $v$ times shorter:

$$\frac{\text{bubble time}}{\text{ideal compute time}} \approx \frac{p-1}{v\,m}.$$

The price is $v$ times as many point-to-point messages. Newer "zero bubble" schedules split each backward into the input-gradient part (which the previous stage waits for) and the weight-gradient part (which nobody waits for), and use the latter to fill most of the bubble.

## Step 8: putting the three together

**3-D parallelism** uses $N = t \times p \times d$ GPUs ($t$ TP size, $p$ stages, $d$ data-parallel replicas), each dimension placed by how much bandwidth it needs, following the Megatron 3-D paper:

~~~dg-layers 3-D parallelism, from the chattiest dimension (fastest links) to the quietest.
*Tensor parallel | inside one node on NVLink; 4 all-reduces per layer, latency-critical
Pipeline parallel | across nodes; one point-to-point send per micro-batch per stage boundary
Data parallel + ZeRO-1 | outermost; one gradient reduction per step, overlapped with the last backwards
~~~

Use just enough pipeline stages to fit the model, since every extra stage adds bubble, and give all remaining GPUs to data parallelism. The rank numbering must match the network: each TP group must be one node, and consecutive pipeline stages should be close. Long-context training adds **context parallelism** (splitting the sequence inside attention) and mixture-of-experts models add **expert parallelism**; the chattiest dimension still gets the fastest links.

## Tensor parallelism in code

Real TP runs one process per GPU. You can check the mathematics on one CPU by keeping shards in a list and replacing each all-reduce by `sum`:

```python
import torch
torch.manual_seed(0); torch.set_num_threads(1)
s, h, t = 6, 8, 4                         # tokens, hidden size, tensor-parallel GPUs
X = torch.randn(s, h, requires_grad=True)
A = torch.randn(h, 4*h) / h**0.5
B = torch.randn(4*h, h) / (4*h)**0.5

Z_full = torch.nn.functional.gelu(X @ A) @ B          # reference: one device

A_sh = A.chunk(t, dim=1)                  # column shards: GPU i holds A_i (h x 4h/t)
B_sh = B.chunk(t, dim=0)                  # row shards:    GPU i holds B_i (4h/t x h)
partial = [torch.nn.functional.gelu(X @ Ai) @ Bi for Ai, Bi in zip(A_sh, B_sh)]
Z_tp = sum(partial)                       # this sum is the all-reduce (operator g)
print((Z_tp - Z_full).abs().max().item()) # 2.98e-07: identical up to rounding

# The wrong split: rows of A, and summing AFTER GeLU
A_rows = A.chunk(t, dim=0); X_cols = X.chunk(t, dim=1)
wrong2 = sum(torch.nn.functional.gelu(Xi @ Ai) for Xi, Ai in zip(X_cols, A_rows)) @ B
print((wrong2 - Z_full).abs().max().item())  # 0.867: GeLU(sum) != sum(GeLU)

# Backward: each GPU gets only a partial dL/dX; operator f all-reduces them
g = torch.randn(s, h)
gX_full, = torch.autograd.grad(Z_full, X, g, retain_graph=True)
gX_parts = [torch.autograd.grad(pz, X, g, retain_graph=True)[0] for pz in partial]
print((sum(gX_parts) - gX_full).abs().max().item())  # 2.38e-07 after summing
print((gX_parts[0] - gX_full).abs().max().item())    # 2.53: one shard alone is wrong
```

The last two lines are why operator $f$ exists: forget the backward all-reduce and the gradient is silently wrong. In production use Megatron-LM's `ColumnParallelLinear` and `RowParallelLinear`, or PyTorch's `torch.distributed.tensor.parallel` with `ColwiseParallel` and `RowwiseParallel` plans, and `torch.distributed.pipelining` for pipeline schedules.

## Checking that it works

- **Match a reference.** Run a 2-layer model for a few steps with and without TP/PP from the same seed and data; fp32 losses should agree to several decimals.
- **Measure the bubble.** In a profile, each stage's idle share should be near $(p-1)/(m+p-1)$; much worse means unbalanced stages.
- **Check communication time.** TP all-reduces should be a minority of each layer's time inside a node; if they dominate, the TP group probably spans nodes.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| TP group spans nodes (wrong rank order or TP > 8) | low MFU, NCCL all-reduce dominates every layer | keep TP within the NVLink domain; check the rank-to-node mapping
| Too few micro-batches ($m$ close to $p$) | 30-50 percent of the step idle | raise $m$ (larger global batch or smaller micro-batch), fewer stages, or interleave
| GPipe schedule with large $m$ | out of memory on the first stage | switch to 1F1B; add activation checkpointing (13.04)
| Unbalanced stages | whole pipeline runs at the slowest stage's speed | balance by time, not layer count; the first and last stages also carry the embedding and the output layer with its loss
| Missing backward all-reduce (operator $f$) | no error, but gradients are wrong and loss lags the reference | test gradients against a single-device reference
| Same dropout seed on every TP rank in sharded regions | correlated dropout masks | separate RNG streams for sharded and replicated regions (Megatron tracks both)
| Heads or vocabulary not divisible by $t$ | shape errors at start-up | choose $t$ dividing heads and KV heads; pad the vocabulary

## Going deeper

**Communication per strategy.** TP sends about $4 \cdot 2\frac{t-1}{t}\,bsh$ elements per GPU per layer per step, a volume that does not shrink with $t$ while compute shrinks as $1/t$: the communication-to-compute ratio grows roughly linearly in $t$. Pipeline traffic is one $bsh$ activation per micro-batch per boundary each way, however many layers a stage holds. Data-parallel traffic is about twice the gradient bytes per step, independent of batch size.

**Why the 1F1B bound holds.** Stage $i$ (counting from 1) never has more than $p - i + 1$ forwards awaiting their backward, because that is how many it can start before the first gradient returns from the last stage. The count does not involve $m$.

**The batch-size squeeze.** $m$ = global batch / ($d$ x micro-batch size), so doubling $d$ halves $m$ and can nearly double the bubble. Pipelines and data parallelism compete for one limited global batch.

**Memory per GPU.** Model state scales as $1/(tp)$, and ZeRO-1 over $d$ divides the optimiser share further. Activations scale as $1/t$ with sequence parallelism; under 1F1B the first stage stores (layers per stage) x $p$ micro-batches, which is roughly independent of $p$.

## Key terms

| Term | Meaning
| --- | --- |
| Tensor parallelism (TP) | Splitting each weight matrix across GPUs so they cooperate on every matmul
| All-reduce | Sum across GPUs with the result on every GPU; four per layer per step under TP
| Sequence parallelism | Splitting norm and dropout regions along the sequence, turning all-reduces into reduce-scatter plus all-gather
| Pipeline parallelism (PP) | Assigning contiguous blocks of layers to stages that pass activations along
| Bubble | Idle stage time; fraction $(p-1)/(m+p-1)$ for GPipe and 1F1B
| 1F1B | Schedule that alternates forwards and backwards, capping stored activations at about $p$ micro-batches
| Interleaved schedule | $v$ layer chunks per GPU, shrinking the bubble about $v$-fold for $v$ times more messages
| 3-D parallelism | TP inside nodes, PP across nodes, DP outermost: $N = t \times p \times d$

<div class="take">
<div class="lab">Key takeaway</div>

split matmuls column-then-row inside a node, pipeline across nodes with enough micro-batches ($m \gg p$) and a 1F1B schedule, and spend the remaining GPUs on data parallelism.

</div>
