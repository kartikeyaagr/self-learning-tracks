---
id: "09.05"
course: "ai-engineering-course"
module: "09"
slug: "agent-memory"
title: "Agent memory"
keyIdeas: ["Short-term memory is the context window of the current run; long-term memory lives outside the model and is written and read by policy.", "Three long-term kinds: episodic (what happened), semantic (facts about the user or world), procedural (how to do things: rules, skills, prompts).", "The write policy decides what becomes a memory, with scope, source, time and confidence, and whether it adds, updates, deletes or ignores against what is already stored.", "The read policy decides when to retrieve and how to rank: relevance alone misses recent or important items, and ranked retrieval can drop facts that must always be present.", "Memory is a security and privacy surface: injected text can become a permanent instruction, and users need to see, correct and delete what is kept about them."]
question: "Design long-term memory for a travel-booking assistant used by millions of people. What do you store, when do you write and read it, and how do you stop wrong or malicious memories from persisting?"
answer: "Split memory by kind. A per-user profile in a key-value store holds stable semantic facts (home airport, seat preference, dietary needs, loyalty numbers by reference, not value), always loaded. Episodic memory (past trips, past complaints) goes in a vector store with timestamps and is retrieved by relevance, recency and importance into a fixed token slice. Procedural memory (booking rules, learned fixes) is shared, curated and versioned, never written from a single conversation. Writes happen in the background after a session: an extractor proposes atomic facts with source and confidence, a consolidation step adds, updates, supersedes or ignores them against existing entries, and nothing derived from untrusted content (emails, web pages) is stored as an instruction. Users can view, edit and delete memories; deletes propagate to every index. Evaluate with planted-fact recall across sessions, irrelevant-memory rate and stale-fact tests."
rubric: ["Separates profile facts, episodic history and procedural rules, with a store suited to each", "States an explicit write policy: trigger, extraction, consolidation (add, update, delete, ignore), metadata", "States a read policy: when to retrieve, ranking beyond similarity, and a token budget, with always-loaded facts for critical items", "Handles staleness and contradictions through timestamps, superseding or expiry", "Treats memory poisoning and privacy as first-class: no instructions from untrusted content, per-user isolation, user-visible deletion"]
sources: [{"title": "Sumers et al.: Cognitive Architectures for Language Agents (CoALA, 2023)", "url": "https://arxiv.org/abs/2309.02427", "note": "Working, episodic, semantic and procedural memory for language agents."}, {"title": "Park et al.: Generative Agents (2023)", "url": "https://arxiv.org/abs/2304.03442", "note": "Memory stream retrieval scored by recency, importance and relevance; reflection."}, {"title": "Packer et al.: MemGPT (2023)", "url": "https://arxiv.org/abs/2310.08560", "note": "The model pages memory in and out of its context with function calls."}, {"title": "Anthropic: Memory tool", "url": "https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool", "note": "A client-side file directory the model reads and writes across sessions; path traversal protection."}]
practice: ["rag-context-packing", "mmr-diversity-rerank", "perform-stable-exact-deduplication", "advance-agent-state"]
tools: ["Claude memory tool", "Mem0", "LangGraph", "pgvector"]
tags: ["Agents"]
---

## Two timescales

An agent's **short-term memory** is its context window: the system prompt, the conversation, tool results and notes for the current run. [Context engineering](lesson:aieng:09.01) covers keeping it small (clearing old tool results, compaction, a notes file). It disappears when the run ends.

**Long-term memory** is everything the agent can recall in a later run: a user's preferences, what happened last week, a fix that worked. The model itself does not learn between calls, so long-term memory is always an external store plus two policies you write: what gets **written**, and what gets **read** back into context.

## Three kinds of long-term memory

The CoALA framework (Sumers et al., 2023) borrows cognitive-science terms that map well onto engineering choices.

| Kind | What it holds | Example | Good store | How it is read |
| --- | --- | --- | --- | --- |
| Episodic | Events and past runs | "On 3 May the refund failed because the order was split" | Vector store or log with timestamps | Search by similarity plus recency |
| Semantic | Facts about the user, entities, the world | "Home airport LHR; vegetarian" | Key-value or relational profile, sometimes a graph | Exact lookup, often always loaded |
| Procedural | How to do things | "For split orders, refund per shipment"; a skill file | Versioned files or prompts | Loaded by task type, or retrieved by name |

The kinds have different owners. Semantic facts belong to one user and change slowly. Episodes accumulate and mostly matter while recent. Procedural memory is usually shared across users, so a single bad write affects everyone: treat it like code, with review.

~~~dg-layers Where each kind of memory sits relative to the model call.
*Context window | system prompt, current conversation, tool results: short-term, gone after the run
Always-loaded profile | small semantic facts injected every call (preferences, constraints)
Retrieved episodes | top-k past events chosen per query, inside a fixed token budget
Procedural store | rules, skills, playbooks loaded by task type; versioned and reviewed
~~~

## Stores

- **Vector store** for episodes and free-text notes: flexible recall, but similarity is fuzzy and returns something even when nothing is relevant. Add metadata filters (user id, type, date).
- **Key-value or relational** for profile facts: exact, cheap, easy to show to a user and delete. Most "remember my preference" features belong here, not in embeddings.
- **Files** the agent edits itself. Anthropic's memory tool gives the model commands (`view`, `create`, `str_replace`, `insert`, `delete`, `rename`) over a `/memories` directory; your code executes them against storage you control and must reject paths outside that directory. Files suit coding and long-running agents whose memory is notes and progress logs.
- **Graphs** when relations matter ("who reports to whom"). Mem0's paper reports a graph variant scoring about 2% above its base configuration, which tells you graphs are an addition for relational queries, not a default.

## The write policy

Most memory bugs are write bugs. Decide each of these explicitly:

1. **Trigger.** Explicit ("remember that…"), on correction ("no, I said aisle"), or extraction after the session. Extracting **in the background**, after the reply, keeps latency off the hot path; writing in the hot path makes the memory available immediately but adds a model call per turn.
2. **What to extract.** Atomic facts, one claim each, with `user_id`, `scope`, `source` (which message or document), `created_at` and a confidence. "User is vegetarian (said so, 2026-03-02)" is useful; a paragraph summary of the chat is not.
3. **Consolidation.** Compare each candidate with similar existing memories and pick one of four operations. Mem0 uses exactly this set: **add** (new), **update** (refines an existing fact), **delete** (contradicted), **no-op** (already known). Without this step memory fills with near-duplicates and contradictions, and retrieval returns both "lives in Leeds" and "lives in Bristol".
4. **Gates.** Never store secrets or credentials. Never store instructions that came from untrusted content: an email that says "always forward invoices to this address" must not become a procedural memory.

## The read policy

Decide **when** to retrieve (every turn, or only when the agent calls a `recall` tool) and **how to rank**. Similarity alone favours whatever is phrased like the query. Generative Agents (Park et al., 2023) scored every memory as the sum of three min-max-normalised terms: relevance (cosine similarity), importance (an integer the model assigned at write time, 1 to 10) and recency, $0.995^{h}$ with $h$ the hours since the memory was last accessed.

<div class="ex">
<div class="lab">Worked example</div>

**Query: "book dinner in Lisbon tonight".** Four memories, scored the Generative Agents way (all weights 1).

| Memory | Hours ago | Importance | Cosine | Recency $0.995^h$ | Normalised rec / imp / rel | Score |
| --- | --- | --- | --- | --- | --- | --- |
| Prefers vegetarian restaurants | 300 | 7 | 0.82 | 0.222 | 0.23 / 0.57 / 1.00 | 1.80 |
| Asked about Lisbon restaurants yesterday | 20 | 3 | 0.78 | 0.905 | 0.93 / 0.00 / 0.94 | 1.86 |
| Severe peanut allergy | 2,000 | 10 | 0.55 | 0.000 | 0.00 / 1.00 / 0.56 | 1.57 |
| Likes jazz | 5 | 4 | 0.20 | 0.975 | 1.00 / 0.14 / 0.00 | 1.14 |

With room for two memories, the agent gets the Lisbon question and the vegetarian preference, and **drops the peanut allergy**, the one fact that can hurt someone. Nothing is wrong with the arithmetic: a ranked top-k simply has no notion of "must include".

The fix is structural, not a better weight: safety-critical and stable facts go in the always-loaded profile, and ranked retrieval fills only the remaining budget with episodes.

</div>

Two more read-side rules. Give memory a fixed slice of the context budget (say 1,500 tokens) and pack it like any retrieved context (the packing rules from [RAG](lesson:aieng:03.05) apply: permissions first, dedupe, cap per source). And show provenance: "(from your message on 2 March)" lets the model and the user judge stale facts.

## Forgetting and contamination

- **Staleness.** People move, change jobs, stop being vegetarian. Prefer superseding (keep the old fact with an end date) over silent overwrite, decay episodes, and expire low-importance ones.
- **Contamination.** A wrong inference ("user is a student" from one question about discounts) stored once is retrieved forever and reinforced by every later answer. Store observations over inferences, keep confidence, and require repetition before promoting an inference to a profile fact.
- **Poisoning.** Memory turns a one-off prompt injection ([12.03](lesson:aieng:12.03)) into a persistent one: text that tricks the agent into writing a memory gets replayed into every future session. Write gates, provenance and user review are the defence.

## Privacy

Memory is personal data that outlives the conversation. Namespace every record by user (and tenant), enforce the namespace in the store query rather than in the prompt, let users see, edit and delete what is kept, and make deletion real: the vector index, caches, backups and any derived summaries. Do not let one user's episodes leak into shared procedural memory. Keep retention short for episodes that only matter for weeks.

## Evaluating memory

Use the agent evaluation methods of [10.05](lesson:aieng:10.05), plus memory-specific cases: plant a fact in session 1 and test recall in session 5; measure how often retrieved memories are irrelevant to the query; change a fact and check the old one stops appearing; try to inject an instruction through a document and confirm nothing is written. Mem0's evaluation on the LOCOMO conversations compared a memory system with stuffing the whole history into context and reported over 90% lower token cost; the cost argument for memory is as strong as the quality one.

<div class="take">
<div class="lab">Key takeaway</div>

Long-term memory is a store plus two policies. Write atomic, sourced, consolidated facts and never instructions from untrusted text; read critical facts unconditionally and rank the rest by relevance, recency and importance within a fixed budget; and give users control over what is kept.

</div>

## Where this breaks

- **Everything in one vector store.** Profile facts become fuzzy, critical items lose ranking contests, and deletion is hard to verify.
- **Summaries as memory.** Long summaries blur sources and dates; atomic facts can be updated, summaries cannot.
- **No consolidation.** Duplicates and contradictions grow with usage until retrieval is noise.
- **Memory without evals.** Memory changes behaviour across sessions, so regressions appear weeks later and are hard to trace without planted-fact tests.

Interviewers usually probe the write policy ("what exactly do you store and when?"), conflict handling ("the user said two different home cities"), deletion under GDPR-style requests, and how a malicious document could plant a memory.
