---
id: "06.05"
course: "inference-eng-course"
module: "06"
slug: "follow-a-request-through-vllm"
title: "Follow a Request Through vLLM"
keyIdeas: ["API server: HTTP, chat template, tokenization, streaming output.", "Engine core: a loop that schedules, runs the model and processes outputs every step.", "Scheduler: picks which requests and how many tokens each run this step, within a token budget.", "KV cache manager: allocates and frees fixed-size blocks.", "Model runner and workers: build batched inputs, run the forward pass and sampling on the GPU."]
question: "Describe the path of a chat request through vLLM from HTTP arrival to the first streamed token."
answer: "The API server receives the request, applies the chat template and tokenizes, and hands it to the engine core as a new request in the waiting queue. On the next step the scheduler allocates KV blocks and schedules its prompt tokens (possibly in chunks). The model runner builds input tensors and runs the forward pass with the attention backend writing K and V into the allocated blocks. The sampler picks the first token, the output processor detokenizes, and the API server streams it as an SSE chunk."
sources: [{"title": "vLLM: Architecture overview", "url": "https://docs.vllm.ai/en/latest/design/arch_overview.html", "note": "Main components and how they connect."}, {"title": "vLLM V1 alpha release blog", "url": "https://blog.vllm.ai/2025/01/27/v1-alpha-release.html", "note": "The redesigned engine core."}]
tags: ["vllm", "internals"]
---

## The pieces of an engine

Every modern engine has the same core components, under different names. This lesson follows vLLM, but SGLang and TensorRT-LLM map onto it closely.

~~~dg-layers The main parts of vLLM, simplified.
API server | HTTP, OpenAI schema, chat template, tokenizer, detokenizer
*Engine core loop | schedule, execute, update outputs, repeat
Scheduler | token budget per step, waiting and running queues, preemption
KV cache manager | block allocation, freeing, prefix cache lookups
Model runner and workers | input preparation, forward pass, sampling, CUDA graphs
~~~

In recent vLLM versions, the API server and the engine core run in separate processes. Tokenization, detokenization and HTTP handling then happen in parallel with GPU work, instead of stealing time from the scheduler loop.

## A request's path, step by step

1. **Arrive.** The API server parses the JSON, validates parameters, applies the chat template and tokenizes the prompt.
2. **Queue.** The request is sent to the engine core and added to the waiting queue.
3. **Schedule.** At the start of the next step, the scheduler fills a token budget, for example 8,192 tokens. Running decode sequences each take one token of the budget first. Waiting requests are then admitted while budget and free KV blocks remain. A long prompt may get only a chunk of the budget this step.
4. **Allocate.** The KV cache manager looks up the prompt's prefix in the prefix cache. Matching blocks are reused and their tokens skipped. New blocks are allocated for the rest.
5. **Execute.** The model runner packs every scheduled token from every request into one flat batch, builds the metadata attention needs (which blocks belong to which sequence, where each sequence starts) and runs one forward pass. Attention reads from and writes to the paged blocks.
6. **Sample.** For each request whose last prompt token or new decode token ran this step, the sampler draws the next token with that request's own settings.
7. **Update and stream.** New tokens are appended, stop conditions are checked, and the text is detokenized and streamed back.
8. **Repeat.** Next step, this request contributes one decode token alongside every other running request, until it finishes and its blocks are freed.

~~~dg-seq One request through vLLM. Press Play to step through it.
API > Core : token ids
Core > Core : schedule, allocate
Core > Runner : flat batch
Runner -> Core : sampled tokens
Core -> API : new tokens
API > API : detokenize, stream
Core > Core : free blocks at end
~~~

## The flat batch

Older systems padded every sequence to the same length and ran a [batch, length] tensor. Modern engines instead concatenate all scheduled tokens into one long list. A step might contain 2,000 prompt tokens from one new request, 500 from another and one token each from 120 decoding requests: 2,620 tokens in total. Matmuls see a [2620, d] input with no padding at all. Attention kernels use per-sequence offsets to know which tokens belong together.

<div class="ex">
<div class="lab">Worked example</div>

**one engine step**

Token budget 4,096. Running: 150 decoding sequences. Waiting: one 6,000-token prompt.

The scheduler gives 150 tokens to the decodes and the remaining 3,946 to the first chunk of the new prompt.

The forward pass processes 4,096 tokens. 150 sequences each gain one output token. The new request's KV cache is filled for its first 3,946 tokens, and no token is sampled for it yet.

Next step: 150 decodes plus the last 2,054 prompt tokens. The new request samples its first token at the end of this step.

</div>

## Why knowing the path helps

Each component maps to a module of this course and to a set of metrics:

- Scheduler: Module 07. Running and waiting queues, preemptions.
- KV cache manager: Module 08. KV usage, block counts.
- Prefix caching and chunked prefill: Module 10. Hit rate, TTFT.
- Model runner, graphs and kernels: Modules 13 to 15. Step time, GPU idle gaps.

When a metric looks wrong, you can ask which component owns it and read that part of the code.

## Deep dive: the flat batch as arrays

The model's forward pass sees one long list of tokens. Metadata says where each sequence starts and how much KV it already has:

~~~dg-raw
%%grid One step's flat batch: three decodes and a 7-token prefill chunk. Each cell is one token in the forward pass.
cols = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
key A = sequence A (decode) | c1
key B = sequence B (decode) | c2
key C = sequence C (decode) | c3
key D = sequence D (prefill chunk) | go
tokens | A B C D D D D D D D
%%
~~~

⟦M482⟧

⟦M483⟧

Every linear layer treats these 10 rows the same. Only attention reads the metadata: row 0 attends to 812 keys of sequence A, row 3 to only 1 key of sequence D, row 9 to 7. This is how one kernel launch can serve a mix of decodes and prefills.

<div class="ex">
<div class="lab">Worked example</div>

**what the batch costs**

The 10-row matmuls cost the same bytes as 1 row: weights dominate. Attention reads ⟦m484⟧ of cached KV for the decodes, about 390 MB for Llama 3 8B, plus a small triangle for the chunk.

So this step's cost is set by the weights (16 GB) and sequence C's long history. Adding sequence D's 7 tokens was almost free.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a request is tokenized, queued, scheduled under a token budget, given KV blocks, packed into a flat batch with everyone else, sampled and streamed. Every engine has these pieces, under different names.

</div>
