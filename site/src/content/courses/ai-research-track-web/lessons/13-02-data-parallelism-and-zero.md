---
id: "13.02"
course: "ai-research-track-web"
module: "13"
slug: "data-parallelism-and-zero"
title: "Data parallelism and ZeRO"
keyIdeas: ["Data parallelism gives every GPU a full replica and a different slice of the batch, then averages gradients with an all-reduce so all replicas take the identical step. Mathematically it is just a larger batch.", "A ring all-reduce is a reduce-scatter followed by an all-gather; each GPU sends and receives about $2(n-1)/n$ times the gradient size, so its bandwidth cost barely depends on the number of GPUs, while its latency term grows with them.", "Plain data parallelism replicates all 16 bytes per parameter on every GPU. ZeRO removes the redundancy in stages: stage 1 shards optimiser states, stage 2 also gradients, stage 3 also the parameters themselves.", "Stages 1 and 2 cost the same communication as ordinary data parallelism; stage 3 costs about 1.5 times as much, because parameters are all-gathered again in the backward pass. PyTorch FSDP is the ZeRO-3 design built into PyTorch.", "Communication is hidden by overlapping it with compute: gradient buckets are reduced while backward continues, and ZeRO-3 prefetches the next layer's parameters while the current layer runs."]
question: "You must fine-tune a 13B model with full Adam on 32 GPUs (4 nodes of 8, 80 GB each, about 50 GB/s per GPU between nodes). Profiling shows activations need about 35 GB per GPU at your chosen micro-batch. Which ZeRO stages fit, and what does each cost in gradient-synchronisation time per optimiser step if nothing is overlapped?"
answer: "Per-GPU model state with $\\Psi = 13$B and $N_d = 32$: plain DP $16 \\times 13 = 208$ GB (impossible); ZeRO-1 $4 \\times 13 + 12 \\times 13/32 = 52 + 4.9 = 56.9$ GB, plus 35 GB of activations is 92 GB, which does not fit; ZeRO-2 $2 \\times 13 + 14 \\times 13 / 32 = 26 + 5.7 = 31.7$ GB, plus 35 GB is about 67 GB, which fits with some headroom for communication buffers and fragmentation; ZeRO-3 $208/32 = 6.5$ GB, plus 35 GB is 41.5 GB, which leaves room for a larger micro-batch. Communication: bf16 gradients are $2 \\times 13 = 26$ GB, so a ring reduce-scatter plus all-gather moves $2 \\times \\frac{31}{32} \\times 26 = 50.4$ GB per rank, about 1.0 s at 50 GB/s for ZeRO-1 or ZeRO-2 (the same as plain DP would cost). ZeRO-3 adds a second parameter all-gather in backward, 1.5 times as much traffic, roughly 75 GB or 1.5 s. ZeRO-2 is the natural choice; pick ZeRO-3 (FSDP `FULL_SHARD`, or `HYBRID_SHARD` so the parameter gathers stay on NVLink inside a node and only gradient reduction crosses the network) if you want the memory for a bigger micro-batch and the step's compute is long enough to hide the extra traffic. Gradient accumulation also helps: synchronisation happens once per optimiser step, not once per micro-batch, so with 4 accumulation steps the 1.0 s is spread over four micro-batches of compute and is far easier to overlap."
sources: [{"title": "Rajbhandari et al.: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2019)", "url": "https://arxiv.org/abs/1910.02054", "note": "Sections 5 to 7: the three stages, their memory formulas and the communication analysis."}, {"title": "Zhao et al.: PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (2023)", "url": "https://arxiv.org/abs/2304.11277", "note": "How FSDP units, prefetching, hybrid sharding and mixed precision are implemented."}, {"title": "The Ultra-Scale Playbook (Hugging Face)", "url": "https://huggingface.co/spaces/nanotron/ultrascale-playbook", "note": "The data-parallelism and ZeRO chapters, with memory and overlap measurements."}]
practice: ["account-for-zero-memory-by-stage", "simulate-distributed-collectives", "collective-effective-bandwidth", "synchronize-data-parallel-gradients", "data-parallel-step-with-psum"]
tags: ["data parallelism", "ZeRO", "FSDP"]
tools: ["PyTorch", "DeepSpeed", "NCCL"]
---

## What you will learn

- How data parallelism splits a batch across GPUs and why it gives the same training run as one big GPU.
- What an all-reduce is, how the ring algorithm does it, and how to estimate its cost.
- Why plain data parallelism wastes most of its memory on identical copies.
- The three ZeRO stages: what each shards, what it saves and what it costs.
- How PyTorch DDP and FSDP map onto these ideas, and how overlap hides communication.

*Before you start:* you need lesson 13.01 (the 16-bytes-per-parameter memory budget and the $6N$ FLOP rule), the Adam optimiser and the idea of a mini-batch gradient.

## From one GPU to many

Lesson 13.01 ended with a budget: a model of $\Psi$ parameters trained with mixed-precision Adam needs about $16\Psi$ bytes before activations and $6\Psi$ FLOPs per token. The FLOPs say one GPU would take years; the memory says a 7.5B model (120 GB of state) does not even fit on an 80 GB card. Data parallelism solves the first problem and ZeRO the second. Tensor and pipeline parallelism (13.03) come in when these run out.

## The idea in one picture

Take four GPUs and a global batch of 128 sequences. Each GPU, called a **rank**, holds a full copy of the model and processes its own 32 sequences. For one weight $w$ the four ranks compute different gradients, because they saw different data:

| rank | 0 | 1 | 2 | 3 | average
| --- | --- | --- | --- | --- | --- |
| examples | 1-32 | 33-64 | 65-96 | 97-128 | all 128
| gradient for $w$ | 0.8 | 1.2 | 0.6 | 1.4 | 1.0

Averaging gives 1.0, exactly the gradient one GPU would compute on all 128 examples (a mean of four equal-sized means is the overall mean). Every rank applies the same update, so the copies stay identical. Turning "each rank has its own number" into "every rank has the average" is an **all-reduce**.

That is **data parallelism (DP)**: same model everywhere, different data, averaged gradients. Its hidden cost is identical copies of every weight, gradient and optimiser moment on every GPU.

~~~dg-fig 13.02-memory.svg
Memory per GPU for a 7.5B model on 64 GPUs. Plain data parallelism keeps all 120 GB on every card. Each ZeRO stage stops replicating one more slice: first the fp32 optimiser state (green), then the gradients (orange), then the weights (blue).
~~~

**ZeRO** (Zero Redundancy Optimizer) keeps the data-parallel recipe but gives each rank only $1/N_d$ of the state it does not need to replicate, where $N_d$ is the number of data-parallel ranks. The bars above show the result: from 120 GB per GPU, which cannot fit, down to 31.4, 16.6 and 1.9 GB.

## Step 1: data parallelism is just a bigger batch

In words: split the global batch into $N_d$ equal **micro-batches**, run forward and backward on each rank, average the gradients, step.

In symbols, with a global batch of $M$ examples split evenly over $N_d$ ranks and per-example loss $\ell_j$,

$$g = \frac{1}{M}\sum_{j=1}^{M} \nabla \ell_j = \frac{1}{N_d}\sum_{r=1}^{N_d} g_r, \qquad g_r = \frac{N_d}{M}\sum_{j \in \text{rank } r} \nabla \ell_j.$$

Here $g$ is the full-batch gradient and $g_r$ is the gradient rank $r$ computes on its own micro-batch. The identity needs equal micro-batch sizes and a loss that is an average over examples. (Batch norm breaks it slightly, since its statistics are per rank.)

To check it, I simulated four ranks in one CPU process: a small two-layer network, 300 SGD steps, batch 128 on one "GPU" versus 4 ranks of 32 with averaged gradients.

~~~dg-fig 13.02-equiv.svg
Left: the single-process run and the four simulated ranks produce the same loss curve. Right: if ranks skip the all-reduce, each trains on its own quarter of the data and the copies drift apart; with the all-reduce they stay bit-identical.
~~~

The two runs ended at the same loss (0.17243) and the largest weight difference between them was $7 \times 10^{-6}$, which is float32 rounding from summing in a different order. Synchronised ranks stayed exactly equal. Without the all-reduce the replicas were 0.02 apart after one step and 0.07 after 300: four different models.

So data parallelism changes the *speed* of training, not the *maths*. The global batch is micro-batch × $N_d$ × accumulation steps, and beyond the critical batch size (13.05) a bigger batch stops buying fewer steps.

## Step 2: the ring all-reduce

How do $N_d$ GPUs average a gradient buffer of $S$ bytes? Sending everything to one rank makes it receive $(N_d - 1)S$ bytes, a bottleneck that worsens with every GPU. The **ring all-reduce** arranges ranks in a circle, cuts the buffer into $N_d$ chunks, and runs two phases:

~~~dg-flow Ring all-reduce as two phases, each n-1 steps around the ring.
Gradients on every rank -(reduce-scatter, n-1 steps)-> *Each rank owns one summed chunk -(all-gather, n-1 steps)-> Every rank has the full sum
~~~

- **Reduce-scatter**: in each step every rank passes one chunk to its right-hand neighbour, which adds it to its own copy. After $n-1$ steps each rank holds the complete sum for one chunk.
- **All-gather**: the finished chunks travel round the ring again, this time copied rather than added. After another $n-1$ steps everyone has every summed chunk.

<div class="ex">
<div class="lab">Worked example</div>

**a ring all-reduce by hand with three GPUs**

Rank 0 holds $(1, 2, 3)$, rank 1 holds $(4, 5, 6)$, rank 2 holds $(7, 8, 9)$. The target is the sum $(12, 15, 18)$ on every rank. Cut each vector into three chunks $c_0, c_1, c_2$ of one number each.

Reduce-scatter, step 1: rank 0 sends its $c_0 = 1$ to rank 1, which now has $c_0 = 4 + 1 = 5$. Rank 1 sends $c_1 = 5$ to rank 2: $8 + 5 = 13$. Rank 2 sends $c_2 = 9$ to rank 0: $3 + 9 = 12$.

Reduce-scatter, step 2: each rank forwards the partial sum it just received. Rank 0 sends $c_2 = 12$ to rank 1: $6 + 12 = 18$. Rank 1 sends $c_0 = 5$ to rank 2: $7 + 5 = 12$. Rank 2 sends $c_1 = 13$ to rank 0: $2 + 13 = 15$.

Now rank 0 owns the finished $c_1 = 15$, rank 1 owns $c_2 = 18$ and rank 2 owns $c_0 = 12$. Two all-gather steps copy these around the ring, and every rank ends with $(12, 15, 18)$.

Traffic: each rank sent 2 chunks in each phase, 4 numbers in total, for a 3-number buffer. That is $\frac{4}{3}S = 2 \cdot \frac{n-1}{n} S$ with $n = 3$.

</div>

In general each rank sends (and receives) $2\frac{n-1}{n}S$ bytes, which approaches $2S$ but never exceeds it, however many GPUs join. A simple time model is

$$t \approx 2(n-1)\,\alpha + 2\,\frac{n-1}{n}\,\frac{S}{B}.$$

Here $n$ is the number of ranks, $\alpha$ the fixed latency of one step, $S$ the buffer size in bytes and $B$ the bandwidth of each link in bytes per second. The bandwidth term barely depends on $n$; the latency term grows with it.

~~~dg-fig 13.02-ring.svg
Left: per-rank traffic of the ring levels off at 2S, while a central rank's load grows with every GPU. Right: the cost model with an assumed 10 µs step latency and 50 GB/s links. Small buffers are dominated by latency and use only a fraction of the link.
~~~

So send big buffers: PyTorch DDP groups gradients into **buckets** (25 MB by default). In this model a 25 MB bucket reaches about 43 GB/s on 8 ranks but only about 22 GB/s on 64, because the latency term has grown; that is why NCCL also uses tree and hierarchical algorithms at scale. In NCCL benchmark output, **bus bandwidth** is the measured $S/t$ times $2(n-1)/n$, so it compares directly with link speed at any GPU count.

## Step 3: where the memory goes

From lesson 13.01, each parameter costs about 16 bytes under mixed-precision Adam:

| item | bytes per parameter | needed by
| --- | --- | --- |
| bf16 weights | 2 | forward and backward
| bf16 gradients | 2 | the optimiser step
| fp32 master weights | 4 | the optimiser step
| fp32 Adam first moment | 4 | the optimiser step
| fp32 Adam second moment | 4 | the optimiser step

The ZeRO paper writes the last three together as $K = 12$ bytes of **optimiser state**. Under plain DP every one of the $N_d$ ranks stores all $16\Psi$ bytes. But the optimiser step is elementwise: updating parameter 17 needs only parameter 17's gradient, master weight and moments. A rank responsible for updating only its $1/N_d$ slice needs optimiser state for that slice alone.

## Step 4: the three ZeRO stages

Each stage stops replicating one more piece of state. With $\Psi$ parameters and $N_d$ ranks:

| stage | what is sharded | memory per GPU | traffic per step
| --- | --- | --- | --- |
| DP (no ZeRO) | nothing | $16\Psi$ | $2\Psi$ (all-reduce)
| ZeRO-1 | optimiser states | $4\Psi + 12\Psi/N_d$ | $2\Psi$
| ZeRO-2 | + gradients | $2\Psi + 14\Psi/N_d$ | $2\Psi$
| ZeRO-3 | + parameters | $16\Psi/N_d$ | $3\Psi$

Memory is in bytes; traffic is in parameter-sized units per rank (multiply by bytes per element, and strictly by $(N_d - 1)/N_d$).

- **ZeRO-1** splits the all-reduce into its halves: a reduce-scatter gives each rank the summed gradient for its shard, the rank runs Adam on that shard, and an all-gather of the updated bf16 weights restores the full model everywhere. Together they *are* an all-reduce, so the saving is free.
- **ZeRO-2** also frees each gradient bucket after its reduce-scatter, keeping only the rank's shard, so gradients drop to $2\Psi/N_d$. Same traffic.
- **ZeRO-3** also shards the bf16 weights and gathers each layer's weights just before use, adding a third collective (Step 5).

~~~dg-fig 13.02-scaling.svg
Memory per GPU as the number of data-parallel GPUs grows, for a 7.5B model. ZeRO-1 and ZeRO-2 level off at the replicated part (30 GB and 15 GB); only ZeRO-3 keeps shrinking as 1/N.
~~~

ZeRO-1 never goes below $4\Psi$ (replicated weights and gradients), ZeRO-2 never below $2\Psi$. For a 70B model those floors are 280 GB and 140 GB, so very large models need ZeRO-3 or the model parallelism of 13.03.

<div class="ex">
<div class="lab">Worked example</div>

**a 7.5B model on 64 GPUs**

Plain DP: $16 \times 7.5 = 120$ GB per GPU. That does not fit on an 80 GB card, before a single activation is stored.

ZeRO-1: $4 \times 7.5 + 12 \times 7.5 / 64 = 30 + 1.41 = 31.4$ GB. ZeRO-2: $2 \times 7.5 + 14 \times 7.5 / 64 = 15 + 1.64 = 16.6$ GB. ZeRO-3: $120 / 64 = 1.88$ GB. These match the figures in the ZeRO paper.

Communication: the bf16 gradient buffer is $2 \times 7.5 = 15$ GB. A ring over 64 ranks moves $2 \times \frac{63}{64} \times 15 = 29.5$ GB per rank per step.

At an assumed 50 GB/s per GPU between nodes (one 400 Gb/s network port per GPU), that is $29.5 / 50 = 0.59$ s per step if nothing overlaps. ZeRO-1 and ZeRO-2 cost the same. ZeRO-3 multiplies it by 1.5: 44.3 GB, about 0.89 s.

If one step's compute takes several seconds, overlap hides most of this. If it takes half a second, the network is now the bottleneck.

</div>

## Step 5: ZeRO-3 and FSDP

Under ZeRO-3 each rank stores only its shard of every layer and borrows the full weights for as long as a layer runs:

~~~dg-seq One ZeRO-3 / FSDP unit (a Transformer block) during a training step.
GPU > Peers: all-gather block parameters (forward)
Peers -> GPU: full parameters; run forward, then free them
GPU > Peers: all-gather block parameters again (backward)
Peers -> GPU: full parameters; run backward, then free them
GPU > Peers: reduce-scatter block gradients
Peers -> GPU: summed gradient for this GPU's shard; local Adam step
~~~

That is three parameter-sized collectives (gather, gather, reduce-scatter) instead of two, hence $3\Psi$ and the 1.5× in the table.

PyTorch's **Fully Sharded Data Parallel (FSDP)** implements this design. The model is wrapped in units, usually one per Transformer block, and the next unit's all-gather is prefetched while the current one computes. Its strategies map onto ZeRO:

- `FULL_SHARD` is ZeRO-3.
- `SHARD_GRAD_OP` keeps parameters gathered between forward and backward, which is close to ZeRO-2.
- `HYBRID_SHARD` shards within a node and replicates across nodes, keeping the frequent all-gathers on fast NVLink.

The newer `fully_shard` API (FSDP2) shards each parameter individually as a DTensor. DeepSpeed exposes the same stages as `zero_optimization.stage`. When even $16\Psi/N_d$ is too much, ZeRO-Offload and ZeRO-Infinity move optimiser states (and optionally parameters) to CPU memory or NVMe, trading PCIe bandwidth for capacity.

## Step 6: hiding communication behind compute

The example assumed nothing overlaps. In reality backward runs from the last layer to the first, so late layers' gradients are ready early, and DDP launches each bucket's all-reduce as soon as it fills while backward continues.

~~~dg-fig 13.02-overlap.svg
A simulated backward pass of eight layers with made-up durations (50 ms of compute and 40 ms of communication per layer). Reducing each bucket as soon as it is ready finishes at 440 ms instead of 720 ms; only the last bucket's reduction is exposed.
~~~

With overlap, step time is roughly $\max(\text{compute}, \text{communication})$ plus the part that cannot overlap. ZeRO-3 overlaps the same way in forward, by prefetching the next unit's weights while the current unit computes.

Gradient accumulation helps too: with four micro-batches per optimiser step, DDP's `no_sync()` skips the all-reduce on the first three, cutting communication per example by four.

## Data parallelism and ZeRO in code

Here is the logic run in one CPU process: a ring all-reduce built from sends between simulated ranks, then a miniature ZeRO step where each rank keeps Adam state for its shard only. The checks confirm that the ring gives the exact sum at $2\frac{n-1}{n}S$ traffic, and that sharded Adam gives the same weights as unsharded Adam.

```python
import numpy as np
rng = np.random.default_rng(0)

def ring_allreduce(bufs):               # bufs: one array per rank
    n = len(bufs); ch = [np.array_split(b.astype(float), n) for b in bufs]; sent = 0
    for phase in ("reduce-scatter", "all-gather"):
        for s in range(n - 1):
            off = -s if phase == "reduce-scatter" else 1 - s
            msgs = [(r, (r + off) % n, ch[r][(r + off) % n].copy()) for r in range(n)]
            for r, c, data in msgs:
                if phase == "reduce-scatter": ch[(r + 1) % n][c] += data
                else: ch[(r + 1) % n][c] = data
            sent += data.size                  # per rank: all send equal amounts
    return [np.concatenate(c) for c in ch], sent

n, S = 8, 1_000_000
grads = [rng.normal(size=S) for _ in range(n)]
out, sent = ring_allreduce(grads)
print(np.allclose(out[3], sum(grads)), sent / S)   # True 1.75  (= 2*(n-1)/n)

def adam(p, g, m, v, t, lr=1e-2, b1=0.9, b2=0.999):
    m[:] = b1*m + (1-b1)*g; v[:] = b2*v + (1-b2)*g*g
    p -= lr * (m / (1-b1**t)) / (np.sqrt(v / (1-b2**t)) + 1e-8)

P = 1000; w_dp = rng.normal(size=P); w_z = w_dp.copy()
m, v = np.zeros(P), np.zeros(P)                     # plain DP: full Adam state
ms, vs = np.zeros((n, P//n)), np.zeros((n, P//n))   # ZeRO: row r lives on rank r
for t in range(1, 51):
    g = sum(rng.normal(size=P) + w_dp for _ in range(n)) / n   # all-reduced gradient
    adam(w_dp, g, m, v, t)
    shards = np.split(w_z.copy(), n)                # rank r updates only shard r
    for r, gr in enumerate(np.split(g, n)):         # gr = its reduce-scatter result
        adam(shards[r], gr, ms[r], vs[r], t)
    w_z = np.concatenate(shards)                    # all-gather the new weights
print(np.abs(w_dp - w_z).max(), ms[0].size)        # 0.0 125
```

On real GPUs you never write the collectives yourself: launch one process per GPU with `torchrun --nproc_per_node=8 train.py`, call `init_process_group("nccl")`, give each rank its own data with a `DistributedSampler`, and wrap the model in `DDP(model)` or apply FSDP2's `fully_shard` to each Transformer block and then to the whole model.

## Checking that it works

- **Replicas agree.** Compare a checksum of the weights across ranks after a few steps. Under DDP they match exactly; a mismatch means a rank skipped a synchronisation.
- **Loss matches a reference.** $N_d$ ranks and one GPU with the same global batch give overlapping curves, as in the CPU test.
- **Memory matches the table.** `torch.cuda.max_memory_allocated()` should be close to the model-state formula plus activations.
- **Communication overlaps.** In a profiler trace, NCCL kernels should run on their own stream alongside backward, not in a block after it.
- **Scaling efficiency.** Tokens per second per GPU at 64 GPUs within about 10 percent of the 8-GPU value means communication is mostly hidden.

## What goes wrong

| symptom | likely cause | fix
| --- | --- | --- |
| Replicas diverge | gradients not synchronised (custom loop, skipped parameters) | use DDP or FSDP; `find_unused_parameters` for skipped branches
| Loss worse after adding GPUs | global batch past the critical batch size; learning rate not retuned | keep the global batch fixed, or retune learning rate and warm-up
| Step time grows with GPU count | exposed communication: small buckets, slow links, no overlap | larger buckets, `HYBRID_SHARD`, accumulation with `no_sync()`
| Out of memory with ZeRO-1/2 at any $N_d$ | replicated floor ($4\Psi$ or $2\Psi$) plus activations exceeds the card | ZeRO-3/FSDP, or activation checkpointing (13.04)
| Hang at a collective | ranks call collectives in different orders, or one crashed | make every rank take the same code path; `NCCL_DEBUG=INFO`

## Going deeper

**Why the ring is bandwidth-optimal.** A reduce-scatter must deliver to each rank the other $n-1$ ranks' contributions to its $S/n$ shard, at least $\frac{n-1}{n}S$ bytes, and an all-gather must deliver the other $n-1$ shards, the same again. The ring meets that $2\frac{n-1}{n}S$ bound exactly. Its weakness is $2(n-1)$ sequential latency steps, which is why tree algorithms and hierarchical schemes (reduce over NVLink inside a node first) win for small messages or many ranks.

**Activations are not sharded.** Each rank still stores the activations of its own micro-batch, so ZeRO is usually paired with activation checkpointing (13.04).

**ZeRO-3 versus tensor parallelism.** Both stop any single GPU from holding all the weights, but they move different things. ZeRO-3 moves *weights* ($3\Psi$ per step, whatever the batch) while every GPU still computes every layer. Tensor parallelism moves *activations* (proportional to batch × sequence × hidden) and splits the computation. Large per-GPU batches favour ZeRO-3; small batches or huge layers favour tensor parallelism.

**Numerics of reduction.** Summation order depends on the algorithm and $N_d$, so results change in the last bits (the $7 \times 10^{-6}$ gap above). Reducing in bf16 over hundreds of ranks can lose small contributions, so gradients are often reduced in fp32.

## Key terms

| Term | Meaning
| --- | --- |
| Rank | One process, normally driving one GPU, in a distributed job
| Data parallelism (DP) | Full model on every rank, different data slices, averaged gradients
| All-reduce | Collective that leaves the sum (or mean) of every rank's buffer on every rank
| Reduce-scatter / all-gather | The halves of an all-reduce: sum and keep one shard each; share all shards
| Bus bandwidth | Measured collective throughput rescaled by $2(n-1)/n$ so it compares directly with link speed
| ZeRO-1 / 2 / 3 | Shard optimiser states / plus gradients / plus parameters across data-parallel ranks
| FSDP | PyTorch's implementation of ZeRO-3-style sharding, with hybrid and gradient-only variants
| Gradient bucket | Gradients reduced together so collectives are large and overlap backward

<div class="take">
<div class="lab">Key takeaway</div>

data parallelism is mathematically a bigger batch kept in sync by one all-reduce of about 2S bytes per rank per step; ZeRO-1 and ZeRO-2 remove the replicated optimiser state and gradients at no extra traffic, and ZeRO-3 or FSDP shards the weights too for 1.5 times the traffic, with overlap hiding most of the cost.

</div>
