---
id: "10.03"
course: "inference-eng-course"
module: "10"
slug: "find-repeated-prefix-computation"
title: "Find Repeated Prefix Computation"
keyIdeas: ["Many requests share prefixes: system prompts, few-shot examples, documents, chat history.", "Each repeated prefix is prefill work that could be skipped.", "Measure: longest common prefix with earlier requests, in tokens, at block granularity.", "The share of repeated prompt tokens bounds the benefit of prefix caching.", "Prompt layout matters: put shared content first, variable content last."]
question: "How would you measure how much prefill work in your traffic is repeated?"
answer: "From logs, tokenize each request's prompt and compute, for each request, the length of the longest prefix matching any earlier request (at block granularity, e.g. 16 tokens). The sum of matched tokens divided by total prompt tokens is the maximum achievable prefix hit rate; it is an upper bound because cache size and eviction limit what is actually kept."
sources: [{"title": "SGLang: Efficient Execution of Structured Language Model Programs", "url": "https://arxiv.org/abs/2312.07104", "note": "Prefix sharing in LLM programs."}, {"title": "vLLM: Automatic prefix caching design", "url": "https://docs.vllm.ai/en/latest/design/prefix_caching.html", "note": "Block-level prefix matching."}]
tags: ["prefix"]
---

## Where repetition comes from

Before turning on prefix caching, find out how much of your prefill work repeats. In many products, most of it does.

- **System prompts.** The same instructions, persona and rules on every request. Often 500 to 3,000 tokens.
- **Tool definitions.** Agents send JSON schemas for every available tool on every call. Often thousands of tokens.
- **Few-shot examples.** Fixed examples placed before the actual question.
- **Multi-turn conversations.** Each turn resends the whole conversation so far. Turn 10 contains turns 1 to 9 again.
- **Retrieval-augmented generation (RAG).** Popular documents are retrieved for many different questions.
- **Agent loops.** Each step resends the task, the tool definitions and the growing scratchpad of earlier steps.

## Measuring repetition from logs

You need token-level matching on prefixes, at block granularity, because that is what the cache can reuse. A trie over token ids does this well:

```
seen = PrefixTrie(block_size=16)
reused = total = 0
for prompt in logs_in_arrival_order:
    ids = tokenize(prompt)
    match = seen.longest_match(ids)        # rounded down to full blocks
    reused += match
    total += len(ids)
    seen.insert(ids)
print("upper bound on prefix reuse:", reused / total)
```

This gives an upper bound, because it assumes an infinite cache that never evicts. A real cache has a finite size, so also simulate an LRU cache of your actual KV pool size. The gap between the two numbers tells you whether memory or prompt layout is the limit.

<div class="ex">
<div class="lab">Worked example</div>

**what a log analysis found**

A support chatbot's logs: 100,000 requests, average prompt 3,100 tokens.

The system prompt is 1,800 tokens. Conversations average 4 turns. Infinite-cache reuse: 81% of prompt tokens.

With an LRU cache the size of one GPU's KV pool: 74%. The difference comes from long conversations being evicted between turns.

Prefill compute could fall by about three quarters. TTFT for a typical request could fall from about 140 ms to about 45 ms.

</div>

## Layout decides what can match

The cache matches exact token prefixes. Any difference, even one token, ends the match, and nothing after it can be reused. So the order of content in the prompt matters as much as the content itself.

| Order of content | What is reused |
| --- | --- |
| user id, system prompt, question | nothing |
| timestamp, system prompt, question | nothing |
| system prompt, user id, question | the system prompt |
| system prompt, tools, documents, history, question | most of the prompt |

Guidelines:

- Put the most stable content first: system prompt, then tool definitions, then semi-stable content such as retrieved documents, then the conversation, then the newest message.
- Move per-request variables such as dates, user names and request ids to the end, or remove them.
- Render templates deterministically. A change in whitespace, key order in JSON tool schemas, or date format breaks the match.

## Common accidental breakers

- A "current time" line at the top of the system prompt, updated every request.
- Tool lists sorted differently on each call, because they came from a dictionary.
- A/B testing frameworks that insert experiment ids into the prompt.
- Chat templates that change how earlier assistant messages are rendered once a new turn arrives, such as stripping reasoning text from old turns.

Each of these can take reuse from 80% to almost 0% without any visible error.

## Deep dive: measuring the ceiling on reuse

Before turning on caching, compute the best hit rate any cache could reach on your logs. For each request, find the longest block-aligned prefix it shares with any earlier request:

⟦M608⟧

LCP is the longest common prefix in tokens, and ⟦m609⟧ is the block size. Real hit rates are lower, because the cache is finite and blocks get evicted, but ⟦m610⟧ tells you whether caching is worth tuning at all.

~~~dg-raw
%%grid Prompt layout decides what can match. Two consecutive turns, good and bad layouts. Each cell is a block.
cols = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
key S = system prompt | c3
key T = tool definitions | c4
key H = conversation history | c2
key U = new user message | c1
key Z = timestamp or request id | wine
good, turn 1 | S S S T T H H U . . . .
good, turn 2 | S S S T T H H H H U . .
bad, turn 1 | Z S S S T T H H U . . .
bad, turn 2 | Z S S S T T H H H H U .
%%
~~~

In the good layout, turn 2 reuses the first 7 blocks. In the bad layout, the timestamp in block 1 changes every block hash after it, and turn 2 reuses nothing.

<div class="take">
<div class="lab">Key takeaway</div>

measure repeated prefixes in real traffic before turning on caching, both with an infinite cache and with your real pool size. Then reorder prompts so stable content comes first and per-request variables come last.

</div>
