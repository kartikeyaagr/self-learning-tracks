---
id: "04.05"
course: "ai-engineering-course"
module: "04"
slug: "what-attention-changes"
title: "What attention changes"
keyIdeas: ["Before transformers, RNNs read text one token at a time and squeezed everything into a fixed-size hidden state: long-range dependencies faded and training could not run in parallel.", "Transformers (2017) dropped recurrence: every token attends directly to every other, in one parallel step, which made training at huge scale and in-context learning possible.", "The price is compute quadratic in context length and a KV cache that grows with every token.", "Linear attention and state-space layers bring back a fixed-size state, now with gating and the delta rule so it forgets and overwrites well.", "2025 hybrids (Qwen3-Next, Nemotron-H, Kimi Linear) use mostly linear layers plus a few full-attention layers for exact recall, cutting long-context memory and decode cost."]
question: "Name two limits of RNNs that attention removed, and one new cost that attention introduced. How do 2025 hybrid models try to get both sides?"
answer: "Attention removed the RNN's fixed-size hidden-state bottleneck: any token is one step from any other, so long-range dependencies do not fade through hundreds of updates. It also removed the sequential training constraint: all positions are processed in parallel, which made training on trillions of tokens on GPUs practical. The new cost is compute that grows with the square of sequence length, plus a KV cache that grows with every token at inference. Hybrids such as Qwen3-Next, Nemotron-H and Kimi Linear make most layers linear attention or state-space layers, which train in parallel but decode with a fixed-size state, and keep a minority of full-attention layers for exact recall. That cuts KV cache and long-context decode cost several-fold while keeping quality."
rubric: ["Names the fixed-size state bottleneck and fading long-range information", "Names sequential training and the lack of parallelism", "Names quadratic compute and the growing KV cache as the new cost", "Explains that hybrids use mostly fixed-state layers plus a few full-attention layers", "Gives why the attention layers are kept (exact recall from anywhere in context)"]
sources: [{"title": "Bahdanau et al.: Neural Machine Translation by Jointly Learning to Align and Translate (2014)", "url": "https://arxiv.org/abs/1409.0473", "note": "Attention added to RNNs, before transformers."}, {"title": "Anthropic: In-context learning and induction heads", "url": "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html", "note": "Attention heads that copy patterns, behind in-context learning."}, {"title": "Yang et al.: Gated Delta Networks (2024)", "url": "https://arxiv.org/abs/2412.06464", "note": "Gating plus the delta rule; the linear layer inside Qwen3-Next and the base of Kimi Linear."}, {"title": "Kimi Team: Kimi Linear (2025)", "url": "https://arxiv.org/abs/2510.26692", "note": "3:1 hybrid of linear and latent attention; up to 75% less KV cache and 6x decode throughput at 1M context."}]
practice: ["parallel-and-recurrent-linear-attention", "gated-deltanet-state-update", "hybrid-attention-capstone", "mamba-2-gated-state-scan", "bahdanau-attention-decoder"]
---

## Before attention

Recurrent neural networks (RNNs, LSTMs) read a sequence one step at a time, updating a hidden state vector. For translation, an encoder RNN read the whole source sentence into one final vector, and a decoder RNN generated the translation from it. Two problems followed:

- **Bottleneck**: a 50-word sentence had to fit in one fixed-size vector, and information from token 1 had to survive every later update to reach token $n$. Long sentences translated badly.
- **Sequential training**: step $t$ needs step $t-1$, so you cannot parallelise over positions. GPUs sat mostly idle.

Bahdanau et al. (2014) let the decoder look back at **all** encoder states at each output step and take a weighted mix of them. Translation of long sentences improved sharply, and the attention weights showed sensible word alignments.

## Attention as the whole model

"Attention Is All You Need" (2017) removed recurrence: every token attends to every other token directly in each layer.

| Aspect | RNN / LSTM | Transformer |
| --- | --- | --- |
| Path between distant tokens | $O(n)$ steps | 1 step |
| Training parallelism | Sequential over positions | Fully parallel |
| Memory of the past | Fixed-size hidden state | All previous tokens (KV cache) |
| Cost per layer | $O(n)$ | $O(n^2)$ |
| Inference state | Constant size | Grows with context |

The figure plots both as sequence length grows: $n$ sequential steps for the RNN against about $n^2/2$ token pairs for causal attention.

~~~dg-raw
%%widget complexity Sequential steps for an RNN versus token pairs for attention, by sequence length.
~~~

Attention does far more total work, but all of it runs in parallel, and GPUs are built for exactly that. That is why the 2017 Transformer trained in 3.5 days on 8 GPUs to beat recurrent models, and why training then scaled to trillions of tokens.

## What it enabled

- **Scale**: parallel training made the data volumes behind scaling laws practical ([How LLMs learn](lesson:aieng:03.02)).
- **Long-range use of context**: any earlier token is one hop away.
- **In-context learning**: Anthropic's interpretability work found "induction heads", attention circuits that spot a pattern earlier in the context ([A][B] ... [A]) and predict what followed it ([B]). They appear at the same point in training as in-context learning ability.

~~~dg-flow An induction head completes a pattern seen earlier in the context.
"Mr Dursley ... Mr" -(attend to token after earlier "Mr")-> *"Dursley" -(copy)-> Predict "Dursley"
~~~

## The quadratic bill

$n^2$ is fine at 2,000 tokens (4 million pairs per head per layer) and painful at 128,000 (16 billion). Doubling a prompt from 32k to 64k tokens doubles the MLP work in prefill but quadruples the attention-score work, so past tens of thousands of tokens time to first token grows faster than the prompt. Three lines of work came out of that bill:

1. **Same maths, less memory traffic**: FlashAttention, still $n^2$ operations but far faster in practice ([FlashAttention](lesson:ieng:14.03)).
2. **Skip or cap pairs**: sliding windows, attention sinks and sparse attention ([The KV cache](lesson:aieng:04.04)).
3. **Replace attention in most layers** with a fixed-size state: linear attention and state-space models, below.

## Linear attention: a fixed-size state again

Drop the softmax and attention can be regrouped. With a feature map $\phi$, causal linear attention is

$$
o_t = \phi(q_t)^\top S_t, \qquad S_t = S_{t-1} + \phi(k_t)\, v_t^\top
$$

where $S_t$ is a $d_k \times d_v$ matrix per head. Training still runs in parallel (in chunks), but decoding is an RNN again: each step updates a state of constant size, with no cache that grows. The catch is the one RNNs always had: a fixed state cannot hold every past token, and plain additive updates never forget, so the state fills with noise.

Two ideas fixed much of that. **Gating** (Mamba-2, gated linear attention) decays the old state, $S_t = \alpha_t S_{t-1} + \dots$, so the model chooses what to forget. The **delta rule** writes by correcting the value currently stored under a key instead of piling on another copy. Gated DeltaNet (Yang et al., 2024) combines both:

$$
S_t = \alpha_t\,(I - \beta_t k_t k_t^\top)\, S_{t-1} + \beta_t\, k_t v_t^\top
$$

Even so, fixed-state layers are weaker at exact recall ("what was the order number on page 3?"), because a lossy summary cannot always return a specific token.

## Hybrid models

The 2025 answer is to mix: mostly linear or state-space layers, plus a few full-attention layers that keep exact recall.

| Model | Linear layers | Full-attention layers | Reported effect |
| --- | --- | --- | --- |
| Qwen3-Next (Alibaba, 2025) | Gated DeltaNet, 3 of every 4 layers | Gated attention, 1 in 4 | 80B total, 3B active MoE; long-context throughput well above dense Qwen3 |
| Nemotron-H (NVIDIA, 2025) | Mamba-2 in most layers | A small minority of self-attention layers | Accuracy on par with Qwen2.5 and Llama 3.1 at up to 3× faster inference |
| Kimi Linear (Moonshot, 2025) | Kimi Delta Attention (Gated DeltaNet with finer gating), 3 of 4 | Multi-head latent attention, 1 in 4 | Beats full MLA under the same recipe; up to 75% less KV cache, 6× decode throughput at 1M context |

The figures are each lab's own reports; check them on your workload. For an app engineer the change shows up as cheaper and faster long-context serving, a KV cache that grows only in the attention layers, and serving support that may lag behind standard transformers (check vLLM or SGLang support before you plan a deployment).

<div class="ex">
<div class="lab">Worked example</div>

**memory for one 128k-token sequence, 32 layers, Llama-3-8B-like shapes**

All attention, 8 KV heads of dimension 128 in bf16: $2 \times 8 \times 128 \times 2 = 4{,}096$ bytes per token per layer, so $32 \times 131{,}072 \times 4$ KiB = 16 GiB.

Hybrid at 3:1: 8 attention layers keep a cache, $8 \times 131{,}072 \times 4$ KiB = 4 GiB. The 24 linear layers each keep one fp32 state per head: $32 \times 128 \times 128 \times 4$ bytes = 2 MiB per layer, 48 MiB in total, and that number is the same at 1k or 1M tokens.

So the hybrid holds a 128k context in about 4 GiB instead of 16, and at 1M tokens the attention layers' 31 GiB is the only part still growing. Decode per token also reads a quarter as much cache, which is where the throughput gains come from.

</div>

## Where this breaks

- **Recall-heavy tasks.** Needle-in-a-haystack retrieval, copying long ids and multi-hop lookups lean on the attention layers; a hybrid with too few of them degrades there first.
- **Short contexts.** At a few thousand tokens the KV cache is small and the savings vanish; the gains are a long-context story.
- **Tooling.** Kernels, quantization recipes and prefix caching for new layer types arrive later than for plain transformers.

<div class="take">
<div class="lab">Key takeaway</div>

Attention replaced a fixed-size memory with direct access to every token, unlocking parallel training and in-context learning at the price of quadratic compute and a growing cache. Hybrid models now bring the fixed-size state back in most layers and keep a few attention layers for exact recall.

</div>
