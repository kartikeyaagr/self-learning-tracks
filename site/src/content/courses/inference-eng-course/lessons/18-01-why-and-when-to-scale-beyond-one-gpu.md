---
id: "18.01"
course: "inference-eng-course"
module: "18"
slug: "why-and-when-to-scale-beyond-one-gpu"
title: "Why and When to Scale Beyond One GPU"
keyIdeas: ["Reason 1: memory. Weights plus KV do not fit on one GPU.", "Reason 2: latency. Splitting each layer across GPUs speeds up each step.", "Reason 3: throughput. More replicas serve more traffic.", "Reason 4: context. Very long sequences need their KV and attention split.", "Prefer the simplest option: replicas if the model fits; tensor parallel within a node if not."]
question: "Give three distinct reasons to use more than one GPU for a model, and the parallelism each typically leads to."
answer: "The model does not fit in one GPU's memory (tensor or pipeline parallelism); you need lower latency per token than one GPU can give, by spreading each layer's work (tensor parallelism); or you need more total throughput than one replica (data parallelism, i.e. more replicas). Very long contexts can also need context parallelism."
sources: [{"title": "Megatron-LM", "url": "https://arxiv.org/abs/1909.08053", "note": "Tensor model parallelism."}, {"title": "vLLM: Parallelism and scaling", "url": "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", "note": "Supported parallel modes."}]
tags: ["parallelism"]
---

## Reasons to use more than one GPU

One GPU is the simplest and most efficient way to serve a model, when it works. There are five distinct reasons to go beyond it, and each points to a different kind of parallelism.

| Reason | Example | Typical choice |
| --- | --- | --- |
| The model does not fit | Llama 3 70B in BF16 (140 GB) on 80 GB GPUs | tensor parallelism of 2 to 4 |
| Per-user latency is too high | an 8B model with a very strict TPOT target | tensor parallelism of 2 |
| Not enough throughput | an 8B model serving ten times more users | data parallelism: more replicas |
| Very long context | prompts of a million tokens | context parallelism |
| A large mixture-of-experts model | hundreds of experts per layer | expert parallelism |

The first and last reasons are about memory. The second is about latency. The third is about throughput. The fourth is about both memory and compute for a single request. Mixing them up leads to poor choices, such as using tensor parallelism to get throughput for a model that would fit on one GPU.

## The cost: communication

Every way of splitting a model across GPUs adds communication:

- **Tensor parallelism** needs an all-reduce of the activations after attention and after the MLP, in every layer, at every step.
- **Pipeline parallelism** passes activations from one stage to the next once per step.
- **Expert parallelism** sends tokens to the GPUs that hold their experts and back, in every MoE layer.
- **Context parallelism** passes keys and values between GPUs during attention.
- **Data parallelism** needs no communication between replicas during a step.

Communication over fast links, such as NVLink inside a server, is cheap enough for tensor parallelism. Over PCIe or between servers, the same communication can erase the gain.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of splitting a model that fits**

An 8B model fits on one 80 GB GPU with about 52 GB of KV pool.

Two replicas on two GPUs: about 2 times the throughput of one, with no communication.

Tensor parallelism of 2 on the same two GPUs: TPOT at low load falls from about 6 ms to about 3.8 ms, because each GPU reads half the weights. But two all-reduces per layer cost time, and total throughput at high load is typically 10 to 25% lower than two replicas.

If the TPOT SLO is 5 ms, only tensor parallelism meets it. If the SLO is 30 ms, two replicas serve more users per GPU.

</div>

## A rule of thumb

1. **If the model and enough KV cache fit on one GPU**, run independent replicas (data parallelism). Add tensor parallelism only if the latency SLO requires it.
2. **If not**, use tensor parallelism inside one server, over NVLink, with the smallest degree that fits the model plus enough KV cache.
3. **Beyond one server**, add pipeline parallelism across servers for dense models, or expert parallelism for MoE models.
4. **For extremely long single requests**, add context parallelism.

These combine. A very large model might use tensor parallelism of 8 inside each server, pipeline parallelism of 2 across two servers, and several such groups as data-parallel replicas.

## Measure, do not assume

The best layout depends on the model, the hardware topology, the workload and the SLO. Later lessons give you the tools to estimate and measure communication costs. The lab at the end of this module compares layouts on one server directly.

## Deep dive: what each GPU holds under tensor parallelism

With TP degree ⟦m894⟧, each GPU holds ⟦m895⟧ of every weight matrix and ⟦m896⟧ of the KV heads:

⟦M893⟧

Llama 3 70B has 8 KV heads, so TP up to 8 splits the KV cleanly. At TP 16, each KV head must be stored on two GPUs, and the KV saving stops at 8.

~~~dg-raw
%%bars Llama 3 70B in BF16 per GPU, by TP degree, with a 1-million-token KV budget across the group (GB).
unit = GB
stack = Weights, KV share
TP 2 | 70+164 | | does not fit on 80 GB
TP 4 | 35+82 | | does not fit on 80 GB
TP 8 | 17.5+41
%%
~~~

TP 2 and TP 4 cannot hold that budget on 80 GB GPUs. TP 8 can, at about 59 GB per GPU plus runtime memory. KV per token for 70B is ⟦m897⟧ KB.

<div class="take">
<div class="lab">Key takeaway</div>

scale out for memory, latency, throughput or context length. Each reason points to a different kind of parallelism, and each adds communication you must measure against the gain.

</div>
