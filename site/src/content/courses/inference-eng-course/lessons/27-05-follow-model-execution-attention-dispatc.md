---
id: "27.05"
course: "inference-eng-course"
module: "27"
slug: "follow-model-execution-attention-dispatch"
title: "Follow Model Execution & Attention Dispatch"
keyIdeas: ["Model runner: build flat input ids and positions for all scheduled tokens.", "Attention metadata: per-request offsets, lengths, block tables, slot mapping.", "Backends implement metadata building and the attention forward.", "CUDA graphs are used when the batch shape matches a captured size.", "Sampling runs on the last position of each request."]
question: "How does the model runner turn a scheduled batch into one forward pass, and where does the attention backend get per-request information?"
answer: "It flattens all scheduled tokens across requests into one input-id tensor with matching positions, builds metadata (query start offsets per request, sequence lengths, block tables, slot mappings for where new KV goes), pads to a captured CUDA graph size if applicable, and calls the model. Each attention layer reads that metadata, built once per step by the backend's metadata builder, to run paged attention over each request's blocks."
sources: [{"title": "vLLM: Attention backends", "url": "https://docs.vllm.ai/en/latest/design/attention_backends.html", "note": "Backend structure."}, {"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "Model runner source."}]
tags: ["internals", "attention"]
---

## Anatomy of a step

~~~dg-flow One model runner step.
Scheduled batch -> Flat input ids and positions -> *Attention metadata (offsets, block tables, slots) -> Forward (graph or eager) -> Logits at last positions -> Sampler
~~~

The model runner turns the scheduler's output into tensors. All tokens from all requests become one flat sequence. The attention metadata carries the per-request structure.

## Metadata fields

| Field | Meaning |
| --- | --- |
| query_start_loc | where each request's tokens begin in the flat batch |
| seq_lens | total context length per request, including cached tokens |
| block_table | the physical blocks of each request |
| slot_mapping | the KV slot where each new token is written |

<div class="ex">
<div class="lab">Worked example</div>

**metadata for a mixed step**

Two decoding requests, R1 at context 900 and R2 at context 40, plus a 100-token prefill chunk for R3, which starts from 0.

Flat input: 1 + 1 + 100 = 102 tokens. query_start_loc = [0, 1, 2, 102]. seq_lens = [900, 40, 100].

With 16-token blocks, R1's new token is at position 899, which is in its block index 56 at offset 3. If that block is physical block 412, its slot is 412 × 16 + 3 = 6,595.

Logits are taken only at positions 0, 1 and 101: the last token of each request. The other 99 positions of R3 produce no sampled token.

</div>

## Dispatch

The selected attention backend (FlashAttention, FlashInfer, Triton and others) provides a metadata builder and an attention implementation. Model layers call a generic attention op that forwards to the backend with this metadata. Switching backends changes the kernel, not the model code.

## Exercise

Log the metadata for a step with two decode requests and one prefill chunk. Check that the offsets and slot mappings match what you calculate by hand.

<div class="take">
<div class="lab">Key takeaway</div>

the model runner flattens a mixed batch into one forward pass, and the per-request structure lives in attention metadata. Reading it shows exactly how continuous batching becomes tensors.

</div>
