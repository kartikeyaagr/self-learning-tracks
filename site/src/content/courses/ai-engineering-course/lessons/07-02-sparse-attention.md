---
id: "07.02"
course: "ai-engineering-course"
module: "07"
slug: "sparse-attention"
title: "Sparse attention"
keyIdeas: ["Full attention scores every pair of tokens: O(n^2). Sparse attention scores only a chosen subset.", "Sliding window (local) attention: each token attends to the last W tokens. Cost O(n x W), and KV cache capped at W.", "Global tokens: a few tokens attend to and are attended by everything (Longformer).", "Strided / dilated patterns: attend to every k-th token to reach further at the same cost (Sparse Transformers).", "Stacking layers widens the receptive field: L layers of window W can pass information about L x W tokens.", "Recent work learns which blocks to attend to (DeepSeek NSA) or mixes local and full-attention layers (Gemma, Mistral)."]
question: "With a sliding window of 4,096 tokens, can a token in layer 30 of a 32-layer model use information from 50,000 tokens back? Explain."
answer: "In principle yes: each layer passes information one window back, so after ~30 layers the receptive field is about 30 x 4,096 = ~123k tokens, which covers 50,000. But the information must be relayed indirectly through about 13 window hops, getting compressed at each, so precise recall from that far back is unreliable. Direct attention only reaches 4,096 tokens per layer; that's why models mix in full-attention layers."
sources: [{"title": "Child et al.: Generating Long Sequences with Sparse Transformers (2019)", "url": "https://arxiv.org/abs/1904.10509", "note": "Strided and fixed sparse patterns."}, {"title": "Beltagy et al.: Longformer (2020)", "url": "https://arxiv.org/abs/2004.05150", "note": "Sliding window plus global tokens."}, {"title": "Yuan et al.: Native Sparse Attention (DeepSeek, 2025)", "url": "https://arxiv.org/abs/2502.11089", "note": "Trainable, hardware-aligned sparse attention."}]
---

## Most pairs don't matter

Full attention compares each token with every previous one. In practice, attention weights are concentrated: most of a token's weight goes to nearby tokens and a handful of distant ones. Sparse attention bets on this and skips the rest, cutting compute and memory from quadratic to closer to linear.

## Common patterns

~~~dg-layers Ways to choose which pairs to score.
*Sliding window | each token sees the last W tokens (Mistral, Gemma local layers)
Dilated / strided | every k-th token further back (Sparse Transformers)
Global tokens | a few tokens see and are seen by all (Longformer, BigBird)
Block / learned selection | pick the most relevant blocks per query (DeepSeek NSA)
~~~

**Sliding window**
Each token attends to its previous W tokens. Cost per token is constant, and the KV cache only needs the last W entries per layer, so memory stops growing with context. Mistral 7B used W = 4,096.

**Strided and dilated**
Child et al.'s Sparse Transformers combined a local pattern with a strided one (every k-th token), so that in two hops any token can reach any other, at O(n sqrt(n)) cost.

**Global tokens**
Longformer adds a few global tokens (like a [CLS] or the question tokens) that attend everywhere. Good for documents where some tokens need the whole picture.

**Learned and block sparsity**
DeepSeek's Native Sparse Attention (2025) compresses distant tokens into block summaries, selects the most relevant blocks for each query, and keeps a local window, all in a form that maps well onto GPU kernels and can be trained end to end.

## Receptive field grows with depth

Information moves one window per layer. Token t in layer 1 sees t - W..t. Those tokens in layer 1 already saw W further back, so in layer 2 token t indirectly sees 2W back. After L layers the theoretical reach is about L x W.

<div class="ex">
<div class="lab">Worked example</div>

**window 4,096, 32 layers**

Reach at layer 30 is about 30 x 4,096 = 122,880 tokens, so 50,000 back is in range in principle.

But the information has to be relayed through about 13 hops, each compressing it, so precise recall of a detail that far back is weak.

That's why many models mix in some full-attention layers, for example Gemma's alternating local and global layers.

</div>

## Trade-offs

| Aspect | Full attention | Sliding window |
| --- | --- | --- |
| Compute per token | O(n) | O(W) |
| KV cache | grows with n | fixed at W per layer |
| Exact long-range recall | Strong | Weak beyond W |
| Kernel support | Universal | Good (FlashAttention supports windows) |

## Explore the patterns

Rows are queries, columns are keys. Pick a pattern and see which pairs are computed. Stack layers to see how far information can travel.

~~~dg-raw
%%widget sparse Sliding-window, dilated, global-token and block-sparse attention patterns.
~~~

How much each pattern saves for a sequence of n tokens:

| Pattern | Pairs computed | Example |
| --- | --- | --- |
| Full causal | about n²/2 | GPT-style models |
| Sliding window w | about n × w | Mistral 7B (w = 4,096) |
| Window + global tokens | n × w + n × g | Longformer, BigBird |
| Interleaved local and global layers | Mix per layer | Gemma 2 and 3 alternate local and global layers |

## Receptive field, worked out

With a window of 4,096 and 32 layers, token information can travel 4,096 positions per layer, so in principle the top layer can see 32 × 4,096 = 131,072 tokens back. In practice information degrades with each hop, so models that need reliable long-range recall keep some full-attention layers. That is the reasoning behind interleaving: most layers cheap and local, a few layers global.

## The serving side

Sliding windows bound the KV cache. A model with a 4,096 window only ever needs the last 4,096 tokens' keys and values per layer, so the cache becomes a fixed-size ring buffer however long the conversation grows. For chat applications with long sessions, that matters more than the FLOP savings.

<div class="take">
<div class="lab">Key takeaway</div>

sparse attention scores only some pairs: a local window, strides, global tokens or learned blocks. Depth stretches reach, but precise long-range recall usually needs some full-attention layers.

</div>
