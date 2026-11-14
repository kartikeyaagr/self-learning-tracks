---
id: "13.02"
course: "ai-research-track-web"
module: "13"
slug: "data-parallelism-and-zero"
title: "Data parallelism and ZeRO"
keyIdeas: ["Data parallelism gives every GPU a full replica and a different slice of the batch, then averages gradients with an all-reduce so all replicas take the identical step. Mathematically it is just a larger batch.", "A ring all-reduce is a reduce-scatter followed by an all-gather; each GPU sends and receives about $2(n-1)/n$ times the gradient size, so its time barely depends on the number of GPUs but is set by the slowest link.", "Plain data parallelism replicates all 16 bytes per parameter on every GPU. ZeRO removes the redundancy in stages: stage 1 shards optimiser states, stage 2 also gradients, stage 3 also the parameters themselves.", "Stages 1 and 2 cost the same communication as ordinary data parallelism; stage 3 costs about 1.5 times as much, because parameters are all-gathered again in the backward pass. PyTorch FSDP is the ZeRO-3 design built into PyTorch.", "Communication is hidden by overlapping it with compute: gradient buckets are reduced while backward continues, and ZeRO-3 prefetches the next layer's parameters while the current layer runs."]
question: "You must fine-tune a 13B model with full Adam on 32 GPUs (4 nodes of 8, 80 GB each, about 50 GB/s per GPU between nodes). Profiling shows activations need about 35 GB per GPU at your chosen micro-batch. Which ZeRO stages fit, and what does each cost in gradient-synchronisation time per optimiser step if nothing is overlapped?"
answer: "Per-GPU model state with $\\Psi = 13$B and $N_d = 32$: plain DP $16 \\times 13 = 208$ GB (impossible); ZeRO-1 $4 \\times 13 + 12 \\times 13/32 = 52 + 4.9 = 56.9$ GB, plus 35 GB of activations is 92 GB, which does not fit; ZeRO-2 $2 \\times 13 + 14 \\times 13 / 32 = 26 + 5.7 = 31.7$ GB, plus 35 GB is about 67 GB, which fits with some headroom for buffers and fragmentation; ZeRO-3 $208/32 = 6.5$ GB, plus 35 GB is 41.5 GB, which leaves room for a larger micro-batch. Communication: bf16 gradients are 26 GB, so a ring reduce-scatter plus all-gather moves $2 \\times \\frac{31}{32} \\times 26 = 50.4$ GB per rank, about 1.0 s at 50 GB/s for ZeRO-1 or ZeRO-2. ZeRO-3 adds a second parameter all-gather, about 1.5 times as much, roughly 1.5 s. ZeRO-2 is the natural choice; pick ZeRO-3 (or FSDP `FULL_SHARD`, or `HYBRID_SHARD` so the parameter gathers stay inside a node) if you want the memory for a bigger micro-batch and the step compute is long enough to hide the extra traffic. Gradient accumulation also helps: synchronisation happens once per optimiser step, not once per micro-batch."
sources: [{"title": "Rajbhandari et al.: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2019)", "url": "https://arxiv.org/abs/1910.02054", "note": "Sections 5 to 7: the three stages, their memory formulas and the communication analysis."}, {"title": "Zhao et al.: PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (2023)", "url": "https://arxiv.org/abs/2304.11277", "note": "How FSDP units, prefetching, hybrid sharding and mixed precision are implemented."}, {"title": "The Ultra-Scale Playbook (Hugging Face)", "url": "https://huggingface.co/spaces/nanotron/ultrascale-playbook", "note": "The data-parallelism and ZeRO chapters, with memory and overlap measurements."}]
practice: ["account-for-zero-memory-by-stage", "simulate-distributed-collectives", "collective-effective-bandwidth", "synchronize-data-parallel-gradients", "data-parallel-step-with-psum"]
tags: ["data parallelism", "ZeRO", "FSDP"]
tools: ["PyTorch", "DeepSpeed", "NCCL"]
---

## Data parallelism

The simplest way to use $n$ GPUs is to give each one a full copy of the model and a different micro-batch. Each rank runs forward and backward on its own data, the gradients are averaged across ranks, and every rank applies the same optimiser step, so the replicas never drift apart. The result is identical, up to floating-point summation order, to training on one GPU with an $n$ times larger batch. That identity is also its limit: you can only grow the batch so far before optimisation stops benefiting (the critical batch size, discussed in the scaling-laws lesson), and every GPU must still hold the whole model.

## The ring all-reduce and what it costs

Averaging gradients is an all-reduce. The bandwidth-optimal ring version splits the buffer of $S$ bytes into $n$ chunks and runs two phases around a ring of GPUs:

~~~dg-flow Ring all-reduce as two phases, each n-1 steps around the ring.
Gradients on every rank -(reduce-scatter, n-1 steps)-> *Each rank owns one summed chunk -(all-gather, n-1 steps)-> Every rank has the full sum
~~~

In each of the $2(n-1)$ steps every rank sends one chunk of size $S/n$ to its neighbour, so per-rank traffic is $2\frac{n-1}{n}S$, close to $2S$ however many GPUs join. A simple cost model is

$$t \approx 2(n-1)\,\alpha + 2\frac{n-1}{n}\frac{S}{B}$$

with per-step latency $\alpha$ and per-link bandwidth $B$. Large buffers are bandwidth-bound and small ones latency-bound, which is why frameworks group gradients into buckets (PyTorch DDP defaults to 25 MB). NCCL's benchmarks report "bus bandwidth", the algorithm bandwidth $S/t$ multiplied by $2(n-1)/n$, so that numbers are comparable across GPU counts and with link speed. Real libraries also use tree and hierarchical algorithms, but the ring formula is the right first estimate.

In practice the all-reduce for one bucket starts as soon as backward has produced it, overlapping with the gradients still being computed. With gradient accumulation you skip synchronisation on all but the last micro-batch (`no_sync()` in DDP).

## Where the memory goes

From lesson 1, mixed-precision Adam needs about 16 bytes per parameter: 2 for bf16 weights, 2 for gradients and 12 for fp32 master weights and moments ($K = 12$ in the ZeRO paper's notation). Data parallelism stores all of it on every rank. ZeRO (Zero Redundancy Optimizer) observes that each rank only needs the full state of the slice it updates, and partitions the rest:

| stage | sharded across $N_d$ ranks | memory per GPU | traffic per step
| --- | --- | --- | --- |
| DP (no ZeRO) | nothing | $16\Psi$ | $2\Psi$ (all-reduce)
| ZeRO-1 | optimiser states | $4\Psi + 12\Psi/N_d$ | $2\Psi$
| ZeRO-2 | + gradients | $2\Psi + 14\Psi/N_d$ | $2\Psi$
| ZeRO-3 | + parameters | $16\Psi/N_d$ | $3\Psi$

Here $\Psi$ is the parameter count and traffic is in parameter-sized units (multiply by bytes per element). ZeRO-1 and ZeRO-2 replace the all-reduce with a reduce-scatter of gradients (each rank gets the summed gradient for its shard), a local optimiser step on that shard, and an all-gather of the updated bf16 weights. A reduce-scatter plus an all-gather is exactly an all-reduce, so the cost is unchanged and the memory saving is free.

## ZeRO-3 and FSDP

Stage 3 also shards the bf16 parameters, so no rank ever holds the whole model at rest. Before a layer runs forward, its parameters are all-gathered, used, and freed; in backward they are gathered again, gradients are computed, and a reduce-scatter sends each rank its shard of the summed gradient. That is three parameter-sized collectives per step instead of two. The extra 50 percent buys memory that scales as $1/N_d$.

~~~dg-seq One ZeRO-3 / FSDP unit (a Transformer block) during a training step.
GPU > Peers: all-gather block parameters (forward)
Peers -> GPU: full parameters; run forward, then free them
GPU > Peers: all-gather block parameters again (backward)
Peers -> GPU: full parameters; run backward, then free them
GPU > Peers: reduce-scatter block gradients
Peers -> GPU: summed gradient for this GPU's shard; local Adam step
~~~

PyTorch's Fully Sharded Data Parallel implements this design. The model is wrapped in units, usually one per Transformer block; each unit's parameters are flattened and sharded, and the next unit's all-gather is prefetched while the current one computes. The sharding strategies map onto ZeRO: `FULL_SHARD` is stage 3, `SHARD_GRAD_OP` keeps parameters gathered between forward and backward (close to stage 2), and `HYBRID_SHARD` shards within a node and replicates across nodes, so the frequent all-gathers stay on fast intra-node links. The newer `fully_shard` API (FSDP2) shards each parameter individually as a DTensor. A mixed-precision policy chooses the parameter dtype for compute and the dtype for gradient reduction; reducing in fp32 is safer for very large $N_d$.

When even $16\Psi/N_d$ is too much, ZeRO-Offload and ZeRO-Infinity move optimiser states (and optionally parameters) to CPU memory or NVMe, trading PCIe bandwidth for capacity. Sharding never reduces activation memory, though; that needs the techniques of the next two lessons.

<div class="ex">
<div class="lab">Worked example</div>

**a 7.5B model on 64 GPUs**

Plain data parallelism: $16 \times 7.5\text{B} = 120$ GB per GPU, which does not fit on an 80 GB card.

ZeRO-1: $4 \times 7.5 + 12 \times 7.5 / 64 = 30 + 1.4 = 31.4$ GB. ZeRO-2: $2 \times 7.5 + 14 \times 7.5/64 = 15 + 1.6 = 16.6$ GB. ZeRO-3: $120/64 = 1.9$ GB. These match the figures in the ZeRO paper.

Communication: gradients in bf16 are $2 \times 6.7\text{B}$ bytes for a 6.7B model, 13.4 GB. A ring over 64 GPUs moves $2 \times \frac{63}{64} \times 13.4 = 26.4$ GB per rank. At an assumed 50 GB/s per GPU across nodes (a 400 Gb/s network port per GPU), that is about 0.53 s per step if nothing overlaps; ZeRO-3 raises it to about 0.79 s. If a step's compute takes several seconds, overlap hides most of it; if it takes 0.5 s, the network becomes the bottleneck.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

data parallelism costs one all-reduce of about $2S$ per rank per step; ZeRO-1 and ZeRO-2 cut the replicated optimiser and gradient memory for free, and ZeRO-3 or FSDP shards everything for 1.5 times the traffic.

</div>
