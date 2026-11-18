---
id: "05.02"
course: "ai-engineering-course"
module: "05"
slug: "multi-head-attention"
title: "Multi-head attention"
keyIdeas: ["Instead of one big attention, split Q, K and V into h smaller heads that attend independently; each has dimension d_model / h (4096 / 32 = 128) and total compute is about the same.", "Head outputs are concatenated back to d_model and mixed by an output projection W_O.", "Some heads specialise (previous token, induction, syntax), but many are redundant.", "Multi-query (MQA) and grouped-query attention (GQA) share K/V across query heads to shrink the KV cache with little quality loss.", "Multi-head latent attention (MLA) caches one compressed latent per token plus a small shared RoPE key and reconstructs every head from it: MQA-like memory with MHA-like quality."]
question: "A model has d_model = 4096 and 32 heads. What is each head's dimension, and how do multi-head outputs get back to size 4096?"
answer: "Head dimension = 4096 / 32 = 128. The input is projected by W_Q, W_K and W_V to seq x 4096 each, then reshaped to (batch, seq, 32, 128) and transposed to (batch, 32, seq, 128), so each of the 32 heads computes its own seq x seq attention and a seq x 128 output. The outputs are transposed back and concatenated to seq x 4096, then multiplied by the output projection W_O (4096 x 4096), which mixes information across heads. Because the heads split the dimension rather than add to it, the parameter count and compute are the same as one 4096-wide head. With grouped-query attention only K and V have fewer heads; the query side and the concatenation are unchanged."
sources: [{"title": "Jay Alammar: The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/", "note": "The multi-head section shows heads attending to different words."}, {"title": "Shazeer: Fast Transformer Decoding, multi-query attention (2019)", "url": "https://arxiv.org/abs/1911.02150", "note": "One shared K/V head to speed up decoding."}, {"title": "Ainslie et al.: GQA (2023)", "url": "https://arxiv.org/abs/2305.13245", "note": "Grouped-query attention and uptraining from MHA checkpoints."}, {"title": "DeepSeek-AI: DeepSeek-V2 (2024)", "url": "https://arxiv.org/abs/2405.04434", "note": "Introduces multi-head latent attention with decoupled RoPE."}]
rubric: ["Computes head dimension 128", "Describes the reshape into 32 heads of 128 and independent attention per head", "Describes concatenation back to 4096", "Names the output projection W_O and its role of mixing heads", "Notes that splitting into heads adds no parameters"]
practice: ["multi-head-latent-attention-mla", "kv-cache-memory-for-mha-mqa-gqa-and-mla", "grouped-query-attention-gqa", "multi-query-attention-mqa", "causal-grouped-query-attention"]
---

## Why more than one head

A single attention head computes one set of weights per token: one way of deciding what matters. But a token often needs several kinds of information at once: the previous word, the subject of the verb, the noun a pronoun refers to. Multi-head attention runs several smaller attentions in parallel, each with its own slice of the Q, K and V projections, so each can specialise.

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right)V_i, \qquad \text{MHA}(X) = [\text{head}_1; \dots; \text{head}_h]\, W_O
$$

with head dimension $d_h = d_\text{model} / h$.

## How it is computed

With $d_\text{model} = 4{,}096$ and $h = 32$ heads, each head works in 128 dimensions.

~~~dg-flow Multi-head attention for one layer.
X (seq x 4096) -(W_Q, W_K, W_V)-> Q, K, V -(split into 32 heads)-> 32 x (seq x 128) -(attention per head)-> *32 head outputs -(concat)-> seq x 4096 -(W_O)-> Output
~~~

```python
# x: (batch, seq, 4096)
q = x @ W_Q                                          # (batch, seq, 4096)
q = q.view(batch, seq, 32, 128).transpose(1, 2)      # (batch, 32, seq, 128)
# same for k and v
out = softmax(q @ k.transpose(-2, -1) / 128**0.5) @ v  # (batch, 32, seq, 128)
out = out.transpose(1, 2).reshape(batch, seq, 4096)  # concatenate the heads
out = out @ W_O                                      # mix information across heads
```

Heads split the dimension rather than add to it, so 32 heads of 128 and one head of 4,096 use the same $W_Q, W_K, W_V, W_O$ and about the same compute. Each head computes its own $\text{seq} \times \text{seq}$ attention matrix; $W_O$ then mixes what the heads found.

## Do heads specialise?

Some do, measurably. Alammar's Illustrated Transformer shows two heads for the word "it": one focused on "animal", another on "tired". Researchers have found heads that attend to the previous token, induction heads that copy from an earlier occurrence of the current token ([What attention changes](lesson:aieng:04.05)), and heads that track syntax. Many heads can also be pruned with little loss, which suggests a lot of redundancy. Treat "head 7 does grammar" stories as partial descriptions, not a design you can rely on.

## Sharing keys and values: MQA and GQA

At decode time the KV cache matters more than compute ([The KV cache](lesson:aieng:04.04)), and MHA caches a key and a value for every head. Shazeer (2019) showed that one shared K/V head for all query heads, **multi-query attention**, makes decoding much faster. It tends to lose some quality and can be less stable to train. **Grouped-query attention** (Ainslie et al., 2023) is the compromise that won: groups of query heads share a K/V head. With 8 KV groups for 32 or 64 query heads, quality stays close to MHA and the cache is 4 to 8 times smaller. Ainslie et al. also showed an existing MHA checkpoint can be converted by mean-pooling its K/V heads and training briefly.

Pick a variant below and see the parameter count and KV cache change.

~~~dg-raw
%%widget heads Multi-head, grouped-query and multi-query attention: weights and KV cache per token.
~~~

## Multi-head latent attention (MLA)

DeepSeek-V2 (2024) took a different route: keep every head distinct, but cache a compressed version of them. Each token's hidden state $h_t$ is projected down to one small latent vector, and per-head keys and values are reconstructed from it:

$$
c_t = W_{DKV}\, h_t, \qquad k_t^{(i)} = W_{UK}^{(i)} c_t, \qquad v_t^{(i)} = W_{UV}^{(i)} c_t
$$

Only $c_t$ is cached (512 values in DeepSeek-V3). Expanding it back for every past token on every step would be slow, so at inference the up-projections are absorbed into the neighbouring matrices: $W_{UK}$ folds into the query projection and $W_{UV}$ into $W_O$, and attention runs directly against the cached latents.

RoPE breaks that trick, because a position-dependent rotation sits between $W_{UK}$ and the query and cannot be folded away. MLA therefore adds a small **decoupled RoPE key** (64 dimensions in DeepSeek-V3) shared by all heads and cached next to the latent; the rotated part carries position, the latent part carries content. MLA is used in DeepSeek-V3 and V3.2 and in Kimi K2, and DeepSeek reported quality at or above MHA.

<div class="ex">
<div class="lab">Worked example</div>

**cache per token per layer, 128 query heads of dimension 128, bf16**

| Variant | What is cached per token per layer | Bytes |
| --- | --- | --- |
| MHA | $2 \times 128$ heads $\times 128$ | 65,536 |
| GQA, 8 groups | $2 \times 8 \times 128$ | 4,096 |
| MQA | $2 \times 1 \times 128$ | 512 |
| MLA (DeepSeek-V3) | latent 512 + RoPE key 64 | 1,152 |

MLA is 57 times smaller than MHA and 3.6 times smaller than 8-group GQA, close to MQA, while every head still gets its own keys and values. DeepSeek-V2's paper puts it as the cache of GQA with only 2.25 groups.

Over DeepSeek-V3's 61 layers and a 128k-token context: MLA needs $1{,}152 \times 61 \times 131{,}072$ bytes, about 8.6 GiB per sequence. The same model with MHA would need about 488 GiB, and with 8-group GQA about 30.5 GiB.

</div>

## Choosing a variant

| Variant | Query heads | K/V cached | KV cache vs MHA | Used in |
| --- | --- | --- | --- | --- |
| MHA | $h$ | $h$ heads | 1× | Original Transformer, GPT-2, Llama 1 |
| GQA | $h$ | $g$ heads (for example 8) | $\tfrac{g}{h}$ | Llama 3, Qwen3, Mistral, gpt-oss, most open models |
| MQA | $h$ | 1 head | $\tfrac{1}{h}$ | PaLM, Falcon |
| MLA | $h$ | one latent + shared RoPE key | about 2 to 4 GQA heads' worth | DeepSeek-V2/V3/V3.2, Kimi K2 |

You do not choose this as an app engineer; the model's authors did. It shows up in what a model costs to serve at long context, and in serving support: MLA needs dedicated kernels, which mainstream engines now ship for DeepSeek-style models.

## Common mistakes

- **Reading head count as capacity.** More heads at the same $d_\text{model}$ means smaller heads, not more parameters.
- **Assuming the KV cache scales with query heads.** It scales with KV heads (GQA, MQA) or the latent size (MLA); read `num_key_value_heads` in the config.
- **Forgetting the transpose.** Reshaping $(\text{batch}, \text{seq}, 4096)$ straight to $(\text{batch}, 32, \text{seq}, 128)$ without the transpose mixes positions across heads, and the code still runs.

<div class="take">
<div class="lab">Key takeaway</div>

Multi-head attention splits $d_\text{model}$ into $h$ heads, runs attention in each, concatenates, and projects with $W_O$. GQA and MQA share keys and values across heads to shrink the cache; MLA caches one compressed latent per token and reconstructs every head from it, getting MQA-like memory with MHA-like quality.

</div>
