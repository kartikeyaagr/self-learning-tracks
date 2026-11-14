---
id: "28.01"
course: "inference-eng-course"
module: "28"
slug: "build-the-model-runner-generation-loop"
title: "Build the Model Runner & Generation Loop"
keyIdeas: ["Implement a Llama-style model yourself or wrap one with control over attention and KV.", "Model runner interface: (token ids, positions, metadata) to last-token logits.", "Load weights from safetensors into your module.", "Generation loop on top: sample, append, stop conditions.", "Validate against Hugging Face outputs."]
question: "What interface should your mini engine's model runner expose so that later lessons can add batching and paging without rewriting it?"
answer: "A function that takes a flat batch of token ids with positions and per-sequence metadata (which tokens belong to which sequence, where their KV lives) and returns logits for the last token of each sequence. Keep the model code free of request logic so the scheduler and KV manager can change independently."
sources: [{"title": "nano-vllm", "url": "https://github.com/GeeeekExplorer/nano-vllm", "note": "A compact engine to compare against."}, {"title": "Hugging Face Llama modeling code", "url": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py", "note": "Reference model."}]
tags: ["mini-engine"]
---

## Structure of the mini engine

Over this module you build a small but real inference engine, one layer at a time.

~~~dg-layers Mini engine components, built over this module.
API | streaming HTTP (28.06)
Engine loop | each step: schedule, run, sample, emit
Scheduler | static then continuous batching (28.03, 28.04)
KV manager | contiguous then paged (28.02, 28.05)
*Model runner | this lesson
~~~

## The model runner interface

```
class ModelRunner:
    def __init__(self, cfg, weights): ...
    def forward(self, input_ids, positions, attn_meta) -> torch.Tensor:  # [num_seqs, vocab]
```

The inputs are tokens, their positions, and metadata that says how they group into sequences. Keep this interface fixed. Every later lesson changes what is behind it (KV cache, batching, paging) without changing how the engine loop calls it.

## Steps

1. **Implement the layers** of a Llama-style model: embedding, RMSNorm, attention with RoPE and GQA, SwiGLU MLP, and the LM head.
2. **Load weights from safetensors,** mapping checkpoint names to your module names.
3. **Compare logits with Hugging Face** for a few prompts. In BF16, expect small differences, so compare with a tolerance and check that the top tokens match.
4. **Write a single-sequence generation loop** with greedy and temperature sampling.

## The core of one layer

```
def layer(x, pos, L):
    h = rmsnorm(x, L.attn_norm)
    q, k, v = h @ L.wq, h @ L.wk, h @ L.wv
    q, k = rope(q, pos), rope(k, pos)
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)   # GQA: share KV heads
    x = x + attention(q, k, v, causal=True) @ L.wo
    h = rmsnorm(x, L.mlp_norm)
    return x + (silu(h @ L.w_gate) * (h @ L.w_up)) @ L.w_down
```

<div class="ex">
<div class="lab">Worked example</div>

**debugging a logit mismatch**

Your top token matches Hugging Face for the first position but not later ones. The likely cause is RoPE: positions off by one, or the wrong pairing of dimensions (interleaved versus split halves).

If every position is wrong by a similar amount, check RMSNorm's epsilon and whether it computes in FP32.

If only long prompts drift, check the RoPE base frequency and any scaling the model's config specifies.

</div>

## Deep dive: how close is close enough

Your engine's logits will not match the reference bit for bit: kernels sum in different orders. Compare distributions, not raw numbers:

⟦M1138⟧

| Check | BF16 implementation, healthy | Likely bug |
| --- | --- | --- |
| Max absolute logit difference | below about 0.1 | above 1 |
| Mean KL per position | below about ⟦m1139⟧ | above ⟦m1140⟧ |
| Top-1 agreement over 1,000 positions | above 99% | below 97% |

A wrong RoPE convention or a missing norm epsilon usually passes the first token and fails later positions, so check across the whole sequence, not only the last one.

<div class="take">
<div class="lab">Key takeaway</div>

start with a clean model runner whose inputs are tokens, positions and metadata. Everything else in the engine builds on that interface.

</div>
