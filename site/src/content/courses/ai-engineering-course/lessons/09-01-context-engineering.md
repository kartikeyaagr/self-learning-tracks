---
id: "09.01"
course: "ai-engineering-course"
module: "09"
slug: "context-engineering"
title: "Context engineering"
keyIdeas: ["Context engineering is choosing everything that goes into the context window on each call: instructions, examples, tool definitions, retrieved data, history and tool results.", "Context is a limited resource: longer contexts cost more, run slower, and quality degrades as length grows ('context rot'). Aim for the smallest set of high-signal tokens that does the job.", "Order for the prompt cache: stable content first (tools, system prompt, examples), growing history after it, anything volatile last. One changing byte near the top makes every call a cache miss.", "For long-running agents: clear stale tool results, compact history, keep state in notes outside the context, and push exploration into sub-agents.", "Clearing and compaction change the prefix, so they invalidate the cache from that point; do them in large, infrequent chunks."]
question: "An agent's quality drops and its cost climbs after 40 tool calls in a long session. Name three context engineering techniques you'd apply, what each fixes, and how you'd keep prompt caching effective while doing them."
answer: "First, clear old tool results: raw file contents and search results from many calls ago add noise and cost, which is context rot; replace them with a short placeholder the agent can re-fetch from. Second, compact: summarise the history into decisions, current state and open issues, and continue from the summary so stale turns stop dominating the window. Third, keep state in structured notes outside the context (a task list or progress file the agent reads and updates), so goals survive clearing and compaction. Sub-agents for exploratory searches and just-in-time retrieval also help. For caching, keep tools, system prompt and examples byte-identical at the front with nothing volatile in them, append history after them, and clear or compact in large chunks at a threshold rather than a little on every call, because each edit invalidates the cached prefix from that point on."
rubric: ["Names at least three techniques (clearing tool results, compaction, external notes, sub-agents, just-in-time retrieval) and what each fixes.", "Ties the quality drop to context rot or distraction from stale content, not only to cost.", "Explains cache-friendly ordering: stable content first, volatile content last, nothing changing in the prefix.", "Recognises that clearing and compaction invalidate the cache after the edit point, so they should be batched."]
sources: [{"title": "Anthropic: Effective context engineering for AI agents", "url": "https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents", "note": "Context as a finite resource; compaction, notes, sub-agents."}, {"title": "Chroma: Context Rot", "url": "https://www.trychroma.com/research/context-rot", "note": "Measured degradation as input length grows."}, {"title": "Liu et al.: Lost in the Middle (2023)", "url": "https://arxiv.org/abs/2307.03172", "note": "Position of information in long contexts matters."}, {"title": "Anthropic: Prompt caching", "url": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching", "note": "Prefix order (tools, system, messages), what invalidates the cache, read and write multipliers."}]
practice: ["rag-context-packing", "prefix-cache-matching-and-reuse", "padding-and-truncation"]
tools: ["Claude API", "OpenAI API"]
---

## From prompts to context

Prompt engineering focused on wording one instruction ([Prompting that holds up](lesson:aieng:03.06)). Agents changed the problem: a model now runs for many turns, and each call's context is a mix of system prompt, tool definitions, history, tool results and retrieved documents. Anthropic's post on context engineering defines the job as curating the optimal set of tokens for each inference, and treats context as a finite resource with diminishing returns.

## Why more context isn't better

- **Cost and latency**: every input token is prefilled and billed, on every call.
- **Context rot**: Chroma's study tested many models and found performance degrading as input length grew, even on simple tasks, well before the context limit.
- **Position effects**: Liu et al. ("Lost in the Middle") placed the one relevant document among up to 30 and moved it around. Accuracy was highest when it was first or last and dropped markedly in the middle, in some settings below the model's accuracy with no documents at all. Newer models handle long contexts better, but irrelevant text is still not free.
- **Attention budget**: irrelevant tokens dilute what matters.

## What goes into a call

~~~dg-layers The parts of an agent's context, roughly in order.
System prompt | role, goals, constraints, at the right altitude: specific but not brittle
Tool definitions | few, distinct, well-described
Examples | a handful of canonical cases, not every edge case
*Retrieved / just-in-time data | fetched when needed, by path or query
History and tool results | trimmed, summarised, stale results removed
~~~

The system prompt should sit at the right altitude: neither hard-coded if-else logic nor vague guidance, organised into sections (background, instructions, tool guidance, output format). Every call resends all of it, and an agent resends it on every step. Set your own numbers:

~~~dg-raw
%%widget context What fills the context window, and what it costs per call and per agent run.
~~~

- **Tool schemas add up.** 50 tools at a few hundred tokens each is over 15,000 tokens before the user has said anything ([Tool calling that holds up](lesson:aieng:08.03) covers tool search).
- **History grows without limit.** In a 100-turn conversation it is the largest part of every call.
- **Agent runs multiply everything.** A 20-step run with 60k tokens of context sends 1.2 million input tokens.

## Order the context for the cache

Prompt caching is what makes agent runs affordable: providers store the processed prefix of a prompt and bill later calls that start with the same bytes at a steep discount. On Anthropic's API cache reads cost 0.1 times the base input price and 5-minute cache writes 1.25 times; OpenAI caches long prompts automatically. Mechanics, breakpoints and TTLs are in [Cost budgeting](lesson:aieng:12.04). For context design one rule matters: **the cache matches a prefix, so anything that changes invalidates everything after it**. Anthropic builds the prefix as tools, then system prompt, then messages.

- Put stable content first: tool definitions in a fixed order, then the system prompt and examples.
- Append, never rewrite: new turns and tool results go at the end.
- Keep volatile values out of the prefix: no timestamps, request ids or per-user details in the system prompt. Put them in the latest user message.
- Don't add, remove or reorder tools mid-run unless you mean to pay for a fresh prefix. (MCP's current spec asks servers to list tools in a deterministic order for exactly this reason.)

<div class="ex">
<div class="lab">Worked example</div>

**one timestamp, six times the bill**

A 20-step agent run: a 10,000-token stable prefix (tools, system prompt, examples), and history that grows by 3,000 tokens per step. Step $i$ sends $10{,}000 + 3{,}000(i-1)$ tokens, so the run sends 770,000 input tokens. Prices are illustrative: \$3 per million input tokens, cache reads at 0.1 times (\$0.30), cache writes at 1.25 times (\$3.75).

- **No caching**: 770,000 × \$3 per million = **\$2.31**.
- **Cache-friendly order**: each step reads the previous step's whole context from cache and writes only the new 3,000 tokens. Reads total 703,000 tokens (\$0.21) and writes 67,000 (\$0.25): **\$0.46**, 80% less.
- **A timestamp at the top of the system prompt**: the prefix never matches, so every token is a cache write: 770,000 × \$3.75 per million = **\$2.89**, 25% more than not caching at all, and about six times the cache-friendly run.

</div>

## Keeping long runs lean

| Technique | How | When |
| --- | --- | --- |
| Tool-result clearing | Replace old tool results with a placeholder | Agents that read files, pages or search results |
| Compaction | Summarise old turns into a short note, keep recent turns verbatim | Long chats and long agent runs |
| Structured notes | The agent writes progress to a file or memory tool and reads it back | Multi-hour tasks, work that spans sessions |
| Just-in-time retrieval | Keep ids or paths in context, fetch content only when needed | Large codebases, document sets |
| Tool search | Load tool definitions on demand instead of all up front | Dozens of tools or more |
| Sub-agents | A helper explores in its own context and returns a short summary | Broad searches that would flood the main context |

**Tool-result clearing** is the cheapest of these. A file the agent read 30 calls ago has already done its job; the agent's own notes about it are what it needs now. Anthropic's API does this server-side (the `clear_tool_uses_20250919` context-editing strategy): once input passes a trigger such as 30,000 tokens, it replaces the oldest tool results with placeholder text, keeps the most recent few, and can exclude tools whose results must stay. You can do the same in your own loop. Keep enough in the placeholder (path, query, a one-line result) that the agent can re-fetch if it needs the content again.

Clearing and compaction both rewrite the prefix, so the cache is invalidated from the first edited message onward. That is why the API exposes a minimum amount to clear per edit: clear in big chunks at a threshold, not a few hundred tokens every step.

**Compaction** summarises the conversation so far (decisions, open issues, key details), drops raw tool output, and continues in a fresh context. **Structured notes** (a to-do list, a NOTES.md) let state survive both; how agents store and retrieve longer-term knowledge is in [Agent memory](lesson:aieng:09.05). **Sub-agents** trade a summary for thousands of tokens of exploration; when that split pays is in [Multi-agent systems](lesson:aieng:09.06).

<div class="ex">
<div class="lab">Worked example</div>

**trimming a coding agent's context**

Before: system prompt 3k, 40 tools 14k, full contents of 25 files 90k, history 30k. 137k tokens per step, slow and expensive, and the agent often edits the wrong file.

After: tools loaded on demand (3k typical), file paths and a grep tool instead of full contents (8k), old tool results cleared and history compacted every 20 turns (10k). About 24k tokens per step.

Input per step drops by 82%, and edits land in the right file more often because the relevant code is no longer buried.

</div>

## Where this breaks

- **Over-aggressive clearing.** If the agent needs a result it no longer has and cannot re-fetch it, it guesses. Keep identifiers in placeholders.
- **Lossy summaries.** A compaction that drops a constraint ("never touch the migrations folder") silently changes behaviour. Keep hard constraints in the system prompt or notes, not only in history.
- **Cache expiry.** Default cache entries live about five minutes. An agent that waits longer between steps (a human approval, a slow build) pays for a fresh write.

<div class="take">
<div class="lab">Key takeaway</div>

Context engineering is curating the smallest set of high-signal tokens for each call. Put stable content first so the cache holds, load data just in time, clear stale tool results and compact in large batches, keep state in notes, and push exploration into sub-agents.

</div>
