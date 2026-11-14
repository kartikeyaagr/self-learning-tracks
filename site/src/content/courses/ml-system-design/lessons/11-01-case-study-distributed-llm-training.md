---
id: "11.01"
course: "ml-system-design"
module: "11"
slug: "case-study-distributed-llm-training"
title: "Case study: distributed LLM training"
keyIdeas: ["Planning a large training run starts from arithmetic: compute ($\\approx 6ND$ FLOPs), memory per GPU, and the time budget decide model size, token count and cluster size.", "Training state does not fit on one GPU, so the design combines data parallelism with sharded optimiser state, tensor parallelism within a node and pipeline parallelism across nodes.", "At thousands of GPUs, failures are routine; frequent asynchronous checkpoints and fast restart matter as much as kernel speed.", "The data pipeline (deduplication, filtering, mixing, tokenisation) and evaluation during training decide quality as much as the model architecture."]
question: "Design the system to pre-train a 70-billion-parameter language model on 15 trillion tokens within about a month. Estimate compute and cluster size, choose a parallelism plan, and describe the data pipeline, checkpointing, failure handling and evaluation during training."
answer: "Compute is about $6ND = 6 \\times 7\\times10^{10} \\times 1.5\\times10^{13} \\approx 6.3\\times10^{24}$ FLOPs. At about 400 TFLOP/s achieved per GPU (40% of a 1 PFLOP/s bf16 peak), that is about 4.4 million GPU-hours, so roughly 8,000 GPUs for 22 days plus margin. Training state with Adam in mixed precision is about 16 bytes per parameter, 1.1 TB, so shard it: tensor parallelism of 8 within each node, pipeline parallelism of 4 to 8 across nodes, and data parallelism with sharded optimiser state over the rest, plus activation checkpointing. Data: deduplicate, filter for quality and safety, mix sources by weight, tokenise ahead of time and stream shards deterministically so restarts resume exactly. Checkpoint asynchronously every 30 to 60 minutes; detect failed nodes, replace from spares and restart automatically. Monitor loss, gradient norm, throughput and MFU, and run small evaluations every few thousand steps. Trade-offs: model size versus tokens, and checkpoint frequency versus overhead."
rubric: ["Estimates total FLOPs with 6ND and converts to GPU-hours and cluster size with a realistic utilisation", "Estimates memory per parameter for training state and concludes sharding is required", "Chooses a parallelism plan matched to the network topology (tensor within node, pipeline and data across)", "Mentions activation checkpointing and the pipeline bubble trade-off", "Designs the data pipeline: deduplication, filtering, mixing and deterministic resumable streaming", "Plans frequent asynchronous checkpoints, failure detection and automatic restart", "Monitors loss spikes, gradient norms and throughput, with a plan for instabilities", "Runs evaluations during training and uses small-scale runs to choose settings"]
sources: [{"title": "Hoffmann et al., Training Compute-Optimal Large Language Models (2022)", "url": "https://arxiv.org/abs/2203.15556", "note": "How to split compute between parameters and tokens."}, {"title": "Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (2021)", "url": "https://arxiv.org/abs/2104.04473", "note": "Combining tensor, pipeline and data parallelism."}, {"title": "Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2019)", "url": "https://arxiv.org/abs/1910.02054", "note": "Sharding optimiser state, gradients and parameters."}, {"title": "Chen et al., Training Deep Nets with Sublinear Memory Cost (2016)", "url": "https://arxiv.org/abs/1604.06174", "note": "Activation checkpointing."}]
practice: ["transformer-training-flop-estimator", "mixed-precision-training-memory-accountant", "choose-topology-aware-parallel-plan", "measure-pipeline-bubble-utilization"]
tags: ["LLM pre-training", "parallelism", "checkpointing", "data pipeline"]
---

## Requirements

Clarify the budget and goal: a 70B dense model, 15T tokens, about a month of wall-clock time, on a cluster of GPUs with 80 GB each, 8 GPUs per node with fast links inside the node and a slower network between nodes. Ask whether the architecture is fixed (assume a standard decoder-only transformer) and what evaluations define success.

## Compute and cluster size

Training FLOPs are about $6ND$, where $N$ is parameters and $D$ tokens:
$6 \times 7\times10^{10} \times 1.5\times10^{13} \approx 6.3\times10^{24}$.
Assume about 40% model FLOPs utilisation on a GPU with roughly 1 PFLOP/s bf16 peak, so 400 TFLOP/s achieved. GPU-seconds are $6.3\times10^{24} / 4\times10^{14} \approx 1.6\times10^{10}$, about 4.4 million GPU-hours. On 8,192 GPUs that is about 22 days, leaving margin for failures and evaluations within a month.

Is 15T tokens sensible? The compute-optimal ratio from scaling-law work is roughly 20 tokens per parameter, which would be 1.4T for 70B. Training far longer than that costs more per unit of quality during training but produces a smaller model that is cheaper to serve, which is usually the right call when the model will serve heavy traffic.

## Memory and parallelism

Mixed-precision training with Adam keeps bf16 weights and gradients, fp32 master weights and two fp32 moments: about 16 bytes per parameter, or 1.1 TB for 70B, before activations. One 80 GB GPU cannot hold it.

- Tensor parallelism (TP) of 8 splits each layer's matrices across the GPUs of a node, using the fast intra-node links for its frequent communication.
- Pipeline parallelism (PP) of 4 to 8 splits layers across nodes. Use enough micro-batches that the pipeline bubble, roughly $(p-1)/m$ for $p$ stages and $m$ micro-batches, stays small.
- Data parallelism (DP) over the remaining factor (8,192 / (8 × 8) = 128 replicas), with optimiser state sharded across replicas (ZeRO stage 1 or 2).
- Activation checkpointing recomputes activations in the backward pass to save memory, at the cost of roughly one extra forward pass, about a third more compute per step.

The global batch is chosen from small-scale runs (a few million tokens per step is typical at this size), with a warmup and a cosine or warmup-stable-decay schedule.

<div class="ex">
<div class="lab">Worked example</div>

**a plan that ignored topology**

A team placed tensor-parallel groups across nodes to balance memory. Every layer then sent large activations over the inter-node network, and throughput fell to under half of the estimate. Regrouping so that each tensor-parallel group sat inside one node, and moving the cross-node split to pipeline stages, recovered the expected utilisation without changing the model.

</div>

## Data pipeline

Deduplicate (exact and near-duplicate with MinHash), filter by quality classifiers and safety rules, remove benchmark contamination, and assign mixture weights per source (web, code, books, maths). Tokenise ahead of time into shards. The loader must be deterministic and resumable: after a restart, it continues from the exact sample, so no data is skipped or repeated.

## Failures and checkpointing

With thousands of GPUs, hardware failures are expected several times a day. A checkpoint of full training state is over 1 TB; write it asynchronously (copy to host memory, then upload) every 30 to 60 minutes. Health checks detect failed GPUs or slow nodes; a controller swaps in spare nodes and restarts from the latest checkpoint automatically. Track "goodput": the fraction of wall-clock time spent on useful training.

## Monitoring and stability

~~~dg-flow Training loop with the main control points.
Tokenised shards -(deterministic loader)-> *Training step on 8,192 GPUs -(loss, grad norm, throughput)-> Metrics and alerts
Training step on 8,192 GPUs -(every 30-60 min, async)-> Checkpoint store -(on failure)-> Restart on spare nodes
Checkpoint store -(every few thousand steps)-> Evaluation jobs
~~~

Watch loss, gradient norm, per-layer activation scales and throughput. Loss spikes are common at scale; responses include skipping the bad batch, rolling back to a checkpoint with a lower learning rate, or architectural safeguards such as QK normalisation. Evaluate on held-out loss and a small benchmark suite regularly to catch regressions early.

## Likely follow-up questions

- "How do you choose hyperparameters without trying them at full size?" Fit scaling curves on small models (learning rate, batch size, loss versus compute) and extrapolate.
- "What if the cluster is shared and nodes come and go?" Use elastic data parallelism and make the parallel plan reconfigurable from a checkpoint.
- "How do you know the run is healthy at day 10?" Compare its loss curve with the scaling-law prediction and with small-run evaluations at the same tokens.

## Trade-offs

- Parameters versus tokens: compute-optimal versus inference-optimal.
- Checkpoint frequency versus lost work and I/O overhead.
- Pipeline depth versus bubble and memory.

<div class="take">
<div class="lab">Key takeaway</div>

plan from arithmetic (6ND, bytes per parameter, utilisation), place tensor parallelism within nodes and pipeline and data parallelism across them, make data loading deterministic and checkpoints asynchronous, and treat failures and loss spikes as routine events with automated responses.

</div>
