---
id: "28.03"
course: "inference-eng-course"
module: "28"
slug: "add-request-batching"
title: "Add Request Batching"
keyIdeas: ["Process multiple sequences per step.", "Linear layers: stack tokens from all sequences.", "Attention: per-sequence lengths; avoid padded attention over empty slots.", "Start with static batching: fixed set of requests until all finish.", "Measure throughput vs batch size."]
question: "How do you batch sequences of different lengths in decode without padding attention for the short ones?"
answer: "Flatten one new token per sequence into a batch for all linear layers, and run attention per sequence (a loop or a variable-length kernel) using each sequence's own cache length. Only attention needs to know lengths; the matmuls just see a [num_seqs, hidden] matrix."
sources: [{"title": "Orca", "url": "https://www.usenix.org/conference/osdi22/presentation/yu", "note": "Selective batching."}, {"title": "PyTorch: scaled_dot_product_attention", "url": "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", "note": "Building block for attention."}]
tags: ["mini-engine", "batching"]
---

## Static batching first

Accept up to B requests, prefill each one, then decode all of them together until each is finished. Finished sequences stop producing tokens, but the batch waits for the longest one.

## Mixed lengths

Requests in a batch have different lengths. The matmuls can still be shared, because each sequence contributes exactly one token per decode step. Only attention needs per-sequence handling.

```
x = embed(next_tokens)                       # [B, d]
for layer in layers:
    q, k, v = proj(x)                        # [B, heads, head_dim]
    write_kv(k, v, positions)
    a = torch.stack([attend(q[i], K[i][:L[i]], V[i][:L[i]]) for i in range(B)])
    x = layer_rest(x, a)
```

A Python loop over sequences is slow but correct. Replace it with a variable-length attention kernel later.

## Measure

| Batch | Tokens/s | TPOT |
| --- | --- | --- |
| 1 | | |
| 8 | | |
| 32 | | |

You should see throughput rise almost linearly at first, as in Module 03, because each step still reads the weights once.

~~~dg-raw
%%widget batch Your measurements should follow this shape: tokens per second rise almost linearly with batch size until compute or KV runs out.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
ctx = 10
b = 3
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the waste in static batching**

A batch of 8 requests with output lengths 50, 60, 80, 90, 120, 150, 300 and 600 tokens.

The batch runs 600 steps. Useful token slots: 1,450. Total slots: 8 × 600 = 4,800.

70% of the batch's capacity is spent on finished sequences. Continuous batching, in the next lesson, fills those slots with new requests.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

batching stacks tokens for the matmuls and keeps attention per sequence. Static batching is the first step; continuous batching comes next.

</div>
