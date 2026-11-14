---
id: "13.03"
course: "ai-research-track-web"
module: "13"
slug: "tensor-and-pipeline-parallelism"
title: "Tensor and pipeline parallelism"
keyIdeas: ["Tensor parallelism splits each weight matrix across GPUs. In Megatron's MLP the first matrix is split by columns and the second by rows, so the only communication is one all-reduce of the layer output in forward and one in backward.", "Each Transformer layer under tensor parallelism needs two all-reduces of a $b \\times s \\times h$ activation in forward and two in backward, on the critical path. That is why tensor parallelism stays inside a node on NVLink.", "Pipeline parallelism assigns contiguous groups of layers to stages and streams micro-batches through them. With $p$ stages and $m$ micro-batches, idle time (the bubble) is $(p-1)/(m+p-1)$ of the step for GPipe and for 1F1B.", "1F1B keeps the same bubble as GPipe but caps in-flight activations at about $p$ micro-batches instead of $m$; interleaving $v$ model chunks per GPU shrinks the bubble by a factor of about $v$ at the price of more point-to-point messages.", "3-D parallelism nests them by bandwidth need: tensor parallel within a node, pipeline across nodes, data parallel (often with ZeRO-1) outermost, with rank order chosen to match the network."]
question: "You have 512 GPUs (64 nodes of 8) for a 70B dense model, a global batch of 512 sequences of 4096 tokens, and micro-batch size 1. Compare layout A (TP 8, PP 8, DP 8) with layout B (TP 8, PP 4, DP 16): number of micro-batches per replica, bubble fraction with 1F1B, and model-state memory per GPU (16 bytes per parameter, before any ZeRO). Why would putting TP 16 across two nodes be a poor third option?"
answer: "Layout A: each of 8 replicas gets 512/8 = 64 sequences, so $m = 64$ and the bubble is $(8-1)/(64+7) = 7/71 \\approx 9.9$ percent. Each GPU holds 1/64 of the model, so state is $70\\text{B} \\times 16 / 64 = 17.5$ GB. Layout B: 16 replicas get 32 sequences each, $m = 32$, bubble $3/35 \\approx 8.6$ percent. Each GPU holds 1/32 of the model: $70 \\times 16 / 32 = 35$ GB, and each stage holds twice as many layers, so its activations (up to about $p$ micro-batches of its layers under 1F1B) are larger too. B has a slightly smaller bubble and more data parallelism but roughly twice the state; it fits only if activations stay under about 40 GB, and ZeRO-1 across the 16 data-parallel ranks can recover most of the optimiser share (12 of the 16 bytes). TP 16 across two nodes would put four all-reduces per layer of a $1 \\times 4096 \\times 8192$ bf16 activation (about 67 MB each) on the inter-node network, on the critical path of every layer, which typically costs far more than the bubble savings. Keep tensor parallelism within the NVLink domain."
sources: [{"title": "Shoeybi et al.: Megatron-LM (2019)", "url": "https://arxiv.org/abs/1909.08053", "note": "Section 3: the column-then-row split of the MLP and attention, with the f and g operators."}, {"title": "Narayanan et al.: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (2021)", "url": "https://arxiv.org/abs/2104.04473", "note": "The interleaved 1F1B schedule and the guidance on combining tensor, pipeline and data parallelism."}, {"title": "Huang et al.: GPipe (2018)", "url": "https://arxiv.org/abs/1811.06965", "note": "Micro-batch pipelining and the bubble analysis that later schedules improve on."}, {"title": "How to Scale Your Model (JAX scaling book)", "url": "https://jax-ml.github.io/scaling-book/", "note": "The chapters on sharded matmuls and Transformer parallelism, with communication costs per strategy."}]
practice: ["reconstruct-column-parallel-mlp-layer", "measure-pipeline-bubble-utilization", "choose-topology-aware-parallel-plan", "tensor-parallel-mlp", "tensor-parallel-communication-cost"]
tags: ["tensor parallelism", "pipeline parallelism", "3-D parallelism"]
tools: ["Megatron-LM", "PyTorch", "NCCL"]
---

## When sharding is not enough

ZeRO-3 shards state, but every GPU still runs the whole model on its own micro-batch: it computes every layer and stores that layer's activations. When one layer's matmuls are too large to run efficiently on one device, when activations dominate memory, or when data parallelism has pushed the global batch past what optimisation tolerates, you need to split the model's computation itself. Tensor parallelism splits within a layer; pipeline parallelism splits across layers.

## Tensor parallelism: Megatron's MLP

Take the MLP $Z = \mathrm{GeLU}(XA)\,B$ with $A$ of shape $h \times 4h$ and $B$ of shape $4h \times h$, on $t$ GPUs. Split $A$ by columns, $A = [A_1, \dots, A_t]$. Each GPU computes $Y_i = \mathrm{GeLU}(XA_i)$ locally: GeLU is elementwise, so no communication is needed (splitting $A$ by rows would need a sum before the nonlinearity). Now split $B$ by rows so that GPU $i$ holds $B_i$ matching its slice of $Y$. Each computes the partial product $Y_iB_i$, and $Z = \sum_i Y_iB_i$ is one all-reduce.

~~~dg-flow Megatron tensor-parallel MLP on one of t GPUs.
X (replicated) -(f: identity forward, all-reduce backward)-> X A_i (column shard) -> GeLU (local) -> Y_i B_i (row shard) -(g: all-reduce forward)-> *Z (replicated)
~~~

The conjugate operators $f$ and $g$ make the backward pass work: $f$ is the identity in forward and an all-reduce of input gradients in backward; $g$ is the reverse. Attention splits the same way: Q, K and V are column-parallel, which assigns whole heads to each GPU, and the output projection is row-parallel. So a layer costs two all-reduces of a $b \times s \times h$ tensor in forward and two in backward. Embeddings and the output layer are split along the vocabulary, with the softmax normaliser computed by a cross-GPU reduction so that full logits are never gathered.

These all-reduces sit on the critical path: the next operation needs their result. With $s = 4096$, $h = 8192$ and bf16, each one is 67 MB per sequence, four times per layer. That is fine at NVLink bandwidth and painful over the inter-node network, so tensor parallelism is normally capped at the GPUs in one node (8). Sequence parallelism (Korthikanti et al.) also splits the norm and dropout regions along the sequence dimension, replacing each all-reduce by a reduce-scatter plus all-gather of the same total volume, so activations outside the matmuls are no longer replicated $t$ times.

## Pipeline parallelism and the bubble

Pipeline parallelism places layers $1..L/p$ on stage 1, the next block on stage 2, and so on. Stages exchange only activations (forward) and activation gradients (backward), point to point, so it tolerates slow links. The cost is idle time. Split the batch into $m$ micro-batches. In GPipe's schedule, every stage runs all $m$ forwards, then all $m$ backwards. Stage $p$ waits $p - 1$ slots before its first forward, and stage 1 waits $p-1$ slots at the end for backwards to drain. Out of $m + p - 1$ slots per phase, $p - 1$ are idle:

$$\text{bubble fraction} = \frac{p-1}{m+p-1}.$$

Papers also quote bubble time relative to ideal compute time, $(p-1)/m$; it is the same idle time measured against a different denominator. Either way, you need $m \gg p$.

GPipe must hold activations for all $m$ micro-batches at once, which fights the large $m$ it needs. The 1F1B schedule (one forward, one backward) runs a short warm-up of forwards, then alternates, starting each micro-batch's backward as early as possible. The bubble is unchanged, but a stage never holds more than about $p$ micro-batches of activations, independent of $m$.

~~~dg-flow Why 1F1B beats GPipe on memory with the same bubble.
[GPipe] F1 F2 ... Fm -> B1 B2 ... Bm -(holds m micro-batches)-> Peak memory grows with m
[1F1B] Warm-up forwards -> *Alternate F and B -(holds about p micro-batches)-> Peak memory fixed by p
~~~

The interleaved schedule from the Megatron 3-D paper gives each GPU $v$ non-contiguous chunks of layers (for example layers 1-2 and 17-18), so each pipeline slot is $v$ times shorter and the bubble-to-compute ratio becomes about $(p-1)/(vm)$. The price is $v$ times as many point-to-point transfers. Newer schedules that split the backward pass into input-gradient and weight-gradient parts can fill most of the remaining bubble.

## Putting the three together

3-D parallelism uses $N = t \times p \times d$ GPUs. The usual guidance, from the Megatron 3-D paper's experiments:

1. Tensor parallel inside a node, up to its GPU count, because it needs the most bandwidth and is latency-critical.
2. Pipeline parallel across nodes, enough stages to fit the model and its activations.
3. Data parallel over the rest, with ZeRO-1 to shard optimiser states; the gradient all-reduce happens once per step and overlaps with the last backward passes.

The ranks must be laid out so each tensor-parallel group is one node and consecutive pipeline stages are close in the network. Long-context training adds context parallelism (splitting the sequence across GPUs inside attention), and mixture-of-experts models add expert parallelism, but the principle is the same: the most chatty dimension gets the fastest links.

<div class="ex">
<div class="lab">Worked example</div>

**bubbles for an 8-stage pipeline**

With $p = 8$, the idle fraction is $7/(m+7)$: $m = 8$ gives $7/15 = 46.7$ percent idle, $m = 16$ gives 30.4 percent, $m = 32$ gives 17.9 percent and $m = 64$ gives 9.9 percent.

Layout: 1,024 GPUs as TP 8 x PP 8 x DP 16. A global batch of 1,024 sequences gives each data-parallel replica 64 sequences; with micro-batch 1 that is $m = 64$ and a 9.9 percent bubble.

Halving the global batch to 512 sequences halves $m$ to 32 and nearly doubles the bubble to 17.9 percent. Interleaving with $v = 2$ at $m = 32$ brings the bubble-to-compute ratio to $7/64 = 0.109$, the same 9.9 percent of the step as $m = 64$ without interleaving.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

split matmuls column-then-row inside a node, pipeline across nodes with enough micro-batches ($m \gg p$) and a 1F1B schedule, and spend the remaining GPUs on data parallelism.

</div>
