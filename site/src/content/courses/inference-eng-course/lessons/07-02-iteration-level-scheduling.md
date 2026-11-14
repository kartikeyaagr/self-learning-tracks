---
id: "07.02"
course: "inference-eng-course"
module: "07"
slug: "iteration-level-scheduling"
title: "Iteration-Level Scheduling"
keyIdeas: ["Schedule at the granularity of one forward pass, not one request.", "After each step: drop finished sequences, admit new ones, decide prompt chunks.", "Orca introduced this, plus \"selective batching\" to batch ops that allow it.", "Attention is per sequence (different lengths), linear layers are batched over all tokens.", "This is the foundation of continuous batching."]
question: "What is the scheduling unit in iteration-level scheduling, and what does that enable?"
answer: "One model iteration (one forward pass that produces one token per running sequence). After every iteration the scheduler can remove finished sequences and add waiting ones, so the batch composition changes step by step, finished slots are refilled immediately, and new requests start without waiting for others to finish."
sources: [{"title": "Orca", "url": "https://www.usenix.org/conference/osdi22/presentation/yu", "note": "Introduced iteration-level scheduling and selective batching."}, {"title": "vLLM: Architecture overview", "url": "https://docs.vllm.ai/en/latest/design/arch_overview.html", "note": "How the engine loop schedules per step."}]
tags: ["scheduling"]
---

## Scheduling iterations, not requests

The key change that made LLM serving efficient came from a 2022 system called Orca. Instead of scheduling whole requests, the server schedules **iterations**: at every forward pass, it decides afresh which sequences take part. A sequence that finished last step is gone. A request that arrived during the last step can join now.

~~~dg-flow The engine loop. Press Play to step through one turn.
Pick sequences -> *Run one forward pass -> Emit tokens -> Remove finished -> Pick sequences
~~~

The loop runs every step, often every 10 to 30 ms. Each turn:

1. Look at the running sequences and the waiting queue.
2. Choose which sequences run this step, and how many tokens each contributes.
3. Run one forward pass for all of them together.
4. Sample a token for each sequence that reached its last scheduled position.
5. Check stop conditions, stream new tokens and remove finished sequences.

## The problem with mixing sequences

Sequences in one step have different lengths and different histories. One has 50 tokens of context, another has 4,000. One is decoding a single token, another is prefilling 512 prompt tokens. How can they share one forward pass without padding?

Orca's answer is **selective batching**, which rests on one observation about the transformer:

- **Linear layers** (the Q, K, V and output projections, the MLP and the LM head) treat every token independently. They do not care which sequence a token belongs to. So you can concatenate all tokens from all sequences into one matrix and run one large matmul.
- **Attention** is the only operation where a token interacts with others, and it must only see its own sequence's history. So attention runs per sequence, or with a kernel that takes variable-length sequences in one launch.

## The flat token list

Modern engines take this to its natural form. All tokens scheduled in a step are packed into one flat list, with metadata that says which sequence each token belongs to and its position.

| Sequence | Tokens this step | Kind | Context after this step |
| --- | --- | --- | --- |
| A | 1 | decode | 812 |
| B | 1 | decode | 2,047 |
| C | 512 | prefill (chunk 1 of 3) | 512 |
| D | 1 | decode | 96 |
| Total | 515 | one forward pass | |

The matmuls see a [515, 4096] input. The attention kernel receives per-sequence start offsets and context lengths, and each sequence's tokens attend only to that sequence's cache. No padding is needed anywhere.

<div class="ex">
<div class="lab">Worked example</div>

**the gain from iteration-level scheduling**

Orca reported up to 36 times higher throughput than a static-batching baseline at the same latency on GPT-3-sized models.

The gain comes from two places. Slots are refilled as soon as a request finishes, so the batch stays full. And new requests start within a step or two, so they no longer wait for whole batches.

</div>

## The scheduler must be fast

Because the scheduler runs on the CPU before every step, its own cost adds directly to step time unless it overlaps with GPU work. With 300 running sequences and decode steps of 15 ms:

- A scheduler that takes 3 ms per step adds 20% to every step.
- Building the attention metadata, block tables and input tensors in Python can take several milliseconds at large batch.

Engines work hard on this loop. Common techniques:

- Keep persistent tensors on the GPU and update them incrementally instead of rebuilding them each step.
- Run the scheduler for step n + 1 while the GPU runs step n (overlap scheduling, used in SGLang and newer vLLM).
- Move tokenization and detokenization to separate processes.

## Where the decisions live

Iteration-level scheduling turns the scheduler into the place where most serving trade-offs are set:

- How many sequences can run at once.
- How many prompt tokens a step may admit.
- Which waiting requests go first.
- What to do when memory runs out.

The next lessons cover each of these.

## Deep dive: the scheduler's time budget

Unless scheduling for step ⟦m495⟧ overlaps with the GPU running step ⟦m496⟧, its CPU time adds to every step:

⟦M494⟧

At batch 256 a decode step takes perhaps 20 ms on the GPU. A Python scheduler that touches every sequence and block table each step can spend 2 to 5 ms. That is 9 to 20% of the GPU idle, at exactly the batch sizes where throughput matters most.

~~~dg-raw
%%timeline Two steps without and with overlapped scheduling (ms).
unit = ms
tick = 5
CPU, serial | 0-4 schedule c4 ; 24-28 schedule c4
GPU, serial | 4-24 step 1 c3 ; 28-48 step 2 c3
CPU, overlapped | 0-4 schedule c4 ; 4-8 schedule next c4
GPU, overlapped | 4-24 step 1 c3 ; 24-44 step 2 c3
%%
~~~

Overlap saves 4 ms every 24, 17% of the run. The price is that the scheduler decides step ⟦m497⟧ before it knows which sequences finished at step ⟦m498⟧, so one finished sequence may run a wasted extra step.

## Deep dive: cost per step must not grow with history

A scheduler that loops over each sequence's full token list does ⟦m499⟧ work per step, which grows as sequences get longer. Engines keep per-sequence counters and incremental block tables so the work is ⟦m500⟧: proportional to the number of sequences, not their lengths.

<div class="take">
<div class="lab">Key takeaway</div>

scheduling per iteration lets the batch change every step. Linear layers batch all tokens together in one flat list, attention keeps each sequence's history separate, and the scheduler must be fast because it runs before every step.

</div>
