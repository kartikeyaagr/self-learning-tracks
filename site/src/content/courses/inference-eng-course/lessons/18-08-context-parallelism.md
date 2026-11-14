---
id: "18.08"
course: "inference-eng-course"
module: "18"
slug: "context-parallelism"
title: "Context Parallelism"
keyIdeas: ["Context parallelism splits one sequence across GPUs along its length.", "Each GPU stores the KV of its chunk; total KV capacity scales with GPUs.", "Ring attention rotates KV blocks between GPUs and overlaps transfer with compute.", "Causal masks make naive splits unbalanced; striped or zig-zag layouts fix it.", "Mainly for long-context prefill; decode with CP needs partial results gathered."]
question: "How does ring attention let several GPUs process one very long prompt, and what does each GPU store?"
answer: "The sequence is split into chunks, one per GPU; each GPU holds Q, K and V for its chunk. GPUs pass K and V blocks around a ring; at each step every GPU computes attention between its queries and the KV block it currently holds, accumulating with online softmax, while sending that block on. After P steps every query has seen all keys. Each GPU stores only its chunk's KV plus one block in transit."
sources: [{"title": "Ring Attention with Blockwise Transformers", "url": "https://arxiv.org/abs/2310.01889", "note": "Ring attention."}, {"title": "Striped Attention", "url": "https://arxiv.org/abs/2311.09431", "note": "Load-balanced ring attention for causal masks."}]
tags: ["context-parallel", "long-context"]
---

## The problem

A single very long request strains one GPU in two ways. A one-million-token prompt on Llama 3 8B needs 1,000,000 × 128 KB = 128 GB of KV cache, more than any single GPU has. And its prefill attention needs on the order of 4 × 32 × 4,096 × (10⁶)² ÷ 2 ≈ 260 PFLOP, which would take several minutes on one H100. Tensor parallelism splits heads and weights, but each GPU still sees every token. **Context parallelism** (CP) splits the tokens themselves.

## Splitting the sequence

Give each of P GPUs a share of the tokens: a contiguous chunk, or a striped set.

- **Linear layers** (projections and MLP) work on each token independently, so each GPU processes its own tokens with no communication, using a full copy of the weights or combined with TP.
- **Attention** is the hard part: each query needs keys and values from every earlier token, which now live on other GPUs.

## Ring attention

Ring attention solves this by passing K and V blocks around a ring of GPUs while each GPU computes attention for its own queries.

~~~dg-flow One pass of ring attention on 4 GPUs.
GPU 0 holds KV 0 -> computes Q0 × KV0 and sends KV0 on -> *receives KV3 -> computes Q0 × KV3 -> after 4 rounds, done
~~~

1. Each GPU starts with its own K and V block and computes attention of its queries against it, keeping the running maximum, denominator and output from online softmax.
2. It sends its current K and V block to the next GPU in the ring and receives one from the previous GPU.
3. It computes attention of its queries against the received block, updating the running values.
4. After P − 1 transfers, every GPU has seen every block, and its output is exact.

Each transfer overlaps with the computation on the previous block. If computing attention on one block takes longer than transferring a block, communication is completely hidden. For long sequences, attention compute grows with the square of the block size while the transfer grows linearly, so the overlap works well.

## Load balance under the causal mask

With a causal mask, a query only attends to earlier tokens. If GPU 0 holds the first quarter of the sequence and GPU 3 the last, GPU 0's queries attend to almost nothing while GPU 3's attend to everything. GPU 3 does about seven times the attention work of GPU 0, and the others wait for it.

The fix is to assign tokens so each GPU has a mix of early and late positions. In **zig-zag** assignment, the sequence is cut into 2P chunks, and GPU i gets chunk i and chunk 2P − 1 − i. Every GPU then has the same total attention work.

<div class="ex">
<div class="lab">Worked example</div>

**CP for a long prefill**

A 256,000-token prompt on Llama 3 8B. One H100 would need 32 GB of KV cache and about 60 s of prefill.

CP 8 across an 8-GPU server: each GPU holds 32,000 tokens and 4 GB of KV. Linear layers run on 32,000 tokens per GPU. Ring attention passes 7 KV blocks of 4 GB each around the ring, overlapped with compute.

Prefill takes about 8 to 9 s instead of 60 s, so TTFT falls by about 7 times, and no GPU needs more than 4 GB of KV for this request.

</div>

## In serving

- **Prefill of very long prompts**: CP shortens TTFT and spreads the KV cache across GPUs.
- **Decode**: the new token's query must attend to KV spread across the GPUs. Each GPU computes a partial attention result over its share, with its local maximum and denominator, and the partials are combined with the log-sum-exp rescaling. Only the small partial results move, not the KV.
- **Combining**: CP is often combined with TP. CP splits the tokens, and TP splits the heads within each CP rank.

## Deep dive: load balance and overlap in ring attention

Split a sequence of ⟦m928⟧ tokens across ⟦m929⟧ GPUs. With contiguous chunks and a causal mask, GPU ⟦m930⟧ must attend to ⟦m931⟧ chunks, so the last GPU does nearly twice the average work. Zigzag partitioning gives GPU ⟦m932⟧ chunks ⟦m933⟧ and ⟦m934⟧, and every GPU does the same work.

~~~dg-raw
%%grid Causal attention work per GPU for 4 GPUs and 8 chunks (rows: query chunks; columns: key chunks). Colors show which GPU owns each query chunk.
cols = k1, k2, k3, k4, k5, k6, k7, k8
key A = GPU 0 | c1
key B = GPU 1 | c2
key C = GPU 2 | c3
key D = GPU 3 | wine
contiguous q1 | A . . . . . . .
contiguous q2 | A A . . . . . .
contiguous q3 | B B B . . . . .
contiguous q4 | B B B B . . . .
contiguous q5 | C C C C C . . .
contiguous q6 | C C C C C C . .
contiguous q7 | D D D D D D D .
contiguous q8 | D D D D D D D D
%%
~~~

Contiguous: GPU 0 does 3 tiles and GPU 3 does 15. With zigzag, GPU 0 owns q1 and q8 (1 + 8 = 9 tiles), GPU 3 owns q4 and q5 (4 + 5 = 9): equal.

The ring also needs communication to hide behind compute. Each step passes one KV chunk of ⟦m935⟧ tokens to the neighbour while attending to the current one:

⟦M927⟧

which grows with ⟦m936⟧. For Llama 3 8B, ⟦m937⟧k on 8 GPUs: 7.3 ms of compute against 0.15 ms to pass 64 MB per layer over NVLink. Communication hides completely.

<div class="take">
<div class="lab">Key takeaway</div>

context parallelism splits a single long sequence across GPUs. Ring attention passes KV blocks around the GPUs while online softmax keeps the result exact, overlapping communication with compute, and zig-zag assignment balances the causal workload.

</div>
