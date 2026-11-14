---
id: "09.01"
course: "ai-engineering-course"
module: "09"
slug: "context-engineering"
title: "Context engineering"
keyIdeas: ["Context engineering is choosing everything that goes into the context window on each call: instructions, examples, retrieved data, tool definitions and results, conversation history.", "Context is a limited resource: longer contexts cost more, are slower, and quality degrades as length grows ('context rot').", "Aim for the smallest set of high-signal tokens that gets the job done.", "Techniques: clear and well-structured system prompts, few canonical examples, minimal non-overlapping tools, retrieval just in time instead of everything up front.", "For long-running agents: compaction (summarise old history), structured notes stored outside the context, and sub-agents that return condensed results.", "Trim tool outputs, put key information at the start or end, and remove stale results."]
question: "An agent's quality drops after 40 tool calls in a long session. Name three context engineering techniques you'd apply and what each fixes."
answer: "1) Clear/trim old tool results: raw outputs from many calls ago add noise and cost, causing context rot. 2) Compaction: summarise history into decisions, current state and next steps and continue in a fresh context, so the window isn't dominated by stale turns. 3) Structured external notes (task list/progress file) the agent reads and updates, so goals and state survive compaction. Also: sub-agents for exploratory sub-tasks that return short summaries, and just-in-time retrieval instead of loading everything."
sources: [{"title": "Anthropic: Effective context engineering for AI agents", "url": "https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents", "note": "Context as a finite resource; compaction, notes, sub-agents."}, {"title": "Chroma: Context Rot", "url": "https://www.trychroma.com/research/context-rot", "note": "Measured degradation as input length grows."}, {"title": "Liu et al.: Lost in the Middle (2023)", "url": "https://arxiv.org/abs/2307.03172", "note": "Position of information in long contexts matters."}]
---

## From prompts to context

Prompt engineering focused on wording one instruction. Agents changed the problem: a model now runs for many turns, calling tools, and each call's context is a mix of system prompt, tool definitions, history, tool results and retrieved documents. Anthropic's post on context engineering defines the job as curating the optimal set of tokens for each inference, and treats context as a finite resource with diminishing returns.

## Why more context isn't better

- **Cost and latency**: every input token is prefilled and billed, on every call.
- **Context rot**: Chroma's study tested many models and found performance degrading as input length grew, even on simple tasks, well before the context limit.
- **Position effects**: "Lost in the Middle" showed information in the middle of a long context is used less reliably.
- **Attention budget**: the model has to spread attention across everything; irrelevant tokens dilute what matters.

## What goes into a call

~~~dg-layers The parts of an agent's context, roughly in order.
System prompt | role, goals, constraints, at the right altitude: specific but not brittle
Tool definitions | few, distinct, well-described
Examples | a handful of canonical cases, not every edge case
*Retrieved / just-in-time data | fetched when needed, by path or query
History and tool results | trimmed, summarised, stale results removed
~~~

## Techniques

**Write a clear system prompt.** Anthropic suggests the right altitude: neither hard-coded if-else logic nor vague guidance. Use sections (background, instructions, tool guidance, output format).

**Load just in time.** Instead of pasting every document, give the agent identifiers (file paths, URLs, queries) and tools to fetch what it needs, the way a person keeps references rather than memorising everything.

**Compaction.** When the context nears its limit, summarise the conversation so far, keep decisions, open issues and key details, drop raw tool output, and continue in a fresh context with the summary.

**Structured notes.** The agent writes progress to a file or memory store (a to-do list, a NOTES.md) and reads it back later, so state survives compaction.

**Sub-agents.** Give a focused sub-task to a separate agent with its own clean context; it returns a short summary instead of thousands of tokens of exploration.

<div class="ex">
<div class="lab">Worked example</div>

**quality drops after 40 tool calls**

Clear old tool results: raw outputs from 30 calls ago are noise. Fixes context rot and cost.

Compact: summarise the session into decisions, current state and next steps, then continue. Fixes the history filling the window.

Externalise state: keep a task list in a notes file the agent updates. Fixes losing track of goals after compaction.

</div>

## Build a context budget

Every call sends the full context: system prompt, tool definitions, history, retrieved documents and room for the answer. In an agent, that is resent on every step. Set your own numbers below.

~~~dg-raw
%%widget context What fills the context window, and what it costs per call and per agent run.
~~~

Things that usually surprise people:

- **Tool schemas add up.** 50 tools at a few hundred tokens each is over 15,000 tokens before the user has said anything.
- **History grows without limit.** A 100-turn conversation can be the largest part of every call.
- **Agent runs multiply everything.** A 20-step run with 60k tokens of context sends 1.2 million input tokens. Prompt caching (04.04) is what makes this affordable.

## Why more context can hurt

Liu et al. (2023, "Lost in the Middle") placed the one relevant document among up to 30 retrieved documents and moved it around. Accuracy was highest when it was first or last and dropped markedly when it was in the middle, in some settings below the model's accuracy with no documents at all. Newer models are better at long contexts, but the rule still holds: irrelevant text is not free. It costs money and dilutes attention.

## Techniques in more detail

| Technique | How | When |
| --- | --- | --- |
| Compaction | Summarise old turns into a short note, keep recent turns verbatim | Long chats and long agent runs |
| Structured notes | The agent writes progress to a file or memory tool and reads it back | Multi-hour tasks, work that spans sessions |
| Just-in-time retrieval | Keep ids or paths in context, fetch content only when needed | Large codebases, document sets |
| Tool search | Load tool definitions on demand instead of all up front | Dozens of tools or more |
| Sub-agents | A helper explores with its own context and returns a short summary | Broad searches that would flood the main context |

<div class="ex">
<div class="lab">Worked example</div>

**trimming a coding agent's context**

Before: system prompt 3k, 40 tools 14k, full file contents of 25 files 90k, history 30k. 137k tokens per step, slow and expensive, and the agent often edits the wrong file.

After: tools loaded on demand (3k typical), file paths and a grep tool instead of full contents (8k), history compacted every 20 turns (10k). About 24k tokens per step.

Cost per step drops by over 80%, and edits land in the right file more often because the relevant code is no longer buried.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

context engineering is curating the smallest set of high-signal tokens for each call. Load data just in time, trim and compact history, keep state in notes, and push exploration into sub-agents.

</div>
