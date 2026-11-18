---
id: "07.06"
course: "ai-engineering-course"
module: "07"
slug: "prompt-rag-fine-tune-or-long-context"
title: "Prompt, RAG, fine-tune or long context?"
keyIdeas: ["Classify failing eval cases before choosing a lever: missing knowledge, ignored context, wrong behaviour or format, missing capability, or too slow and expensive. Each has a different first fix.", "Knowledge goes in context (retrieval or long context); behaviour and format go in prompts first and weights second. Fine-tuning new facts in is slow and raises hallucination.", "Long context is often the best-quality option when the material fits, and prompt caching makes repeated long prefixes about 10x cheaper; RAG stays cheaper per request and is the only option for large, changing or permissioned corpora.", "Cost depends on volume: at thousands of requests a month engineering time dominates; at millions, a fine-tuned small model behind retrieval can cost a tenth or less of a frontier model.", "The strongest production systems combine levers: retrieval for facts, a small fine-tune for format and cost, caching for shared prefixes."]
question: "A support assistant answers from a 100k-token knowledge base that changes monthly. Today it stuffs the whole knowledge base into every prompt of a frontier model. Traffic is growing from 100k to 10M requests a month. Walk through which levers you would consider (prompting, caching, RAG, fine-tuning a smaller model, self-hosting), what each would fix or cost at both volumes, and how you would decide."
answer: "Start from an eval set and classify failures, since cost and quality pull different ways. Knowledge belongs in context, so fine-tuning cannot replace retrieval; it can only change which model reads the context. At 100k requests a month, prompt caching keeps the long-context design: with a byte-identical prefix, reads are a tenth of the input price, about 3,500 dollars a month at Sonnet-class list prices versus about 31,000 uncached, close to RAG's roughly 2,000. If long context answers better on the eval, keep it. At 10M a month even cached long context is about 350,000 dollars, RAG on the frontier model about 195,000, and RAG with a fine-tuned small model about 45,000 hosted or under 10,000 self-hosted. Moving costs a labelled set, a fine-tune trained on teacher outputs with retrieved context, a before-and-after eval per slice, and a plan for retraining when the base model is retired. Choose the cheapest option that passes the eval at each volume."
rubric: ["Diagnoses failure types from an eval before picking a lever", "Keeps knowledge in context and does not propose fine-tuning facts in", "Accounts for prompt caching when costing long context, including the cache staying cold at low volume", "Computes or estimates costs at both volumes and notes where engineering time dominates", "Proposes a combined design (retrieval plus a fine-tuned small model) with an eval gate and a retraining plan"]
sources: [{"title": "Ovadia et al.: Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs (2023)", "url": "https://arxiv.org/abs/2312.05934", "note": "RAG consistently beats unsupervised fine-tuning for both known and new knowledge."}, {"title": "Gekhman et al.: Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? (2024)", "url": "https://arxiv.org/abs/2405.05904", "note": "New facts are learned slowly in fine-tuning and raise hallucination once learned."}, {"title": "Li et al.: Retrieval Augmented Generation or Long-Context LLMs? (2024)", "url": "https://arxiv.org/abs/2407.16833", "note": "Long context wins on quality when resourced; RAG wins on cost; Self-Route mixes them."}, {"title": "Zhang et al.: RAFT: Adapting Language Model to Domain Specific RAG (2024)", "url": "https://arxiv.org/abs/2403.10131", "note": "Fine-tuning a model to use retrieved documents and ignore distractors."}]
practice: ["prompt-cache-cost", "rag-context-packing", "lora-finetune-memory", "optimize-throughput-and-cost-under-latency-slo"]
tools: ["Claude API", "Vertex AI tuning", "vLLM"]
tags: ["Fine-tuning", "RAG", "Cost", "Exercise"]
---

## Diagnose before you choose

[Making a model useful](lesson:aieng:03.04) gave the ladder: prompt, retrieve, fine-tune, preference-tune. In practice the choice is made per failure, so start by classifying the cases your eval set ([building an eval set](lesson:aieng:10.02)) marks as wrong. Five causes cover nearly all of them, and each has a different first fix.

~~~dg-flow Classify failing cases first; each class has its own first lever.
[Missing knowledge] Failing case -(fact not in context)-> *Retrieval or long context
[Ignored context] Failing case -(fact retrieved, not used)-> Better prompt, fewer distractors, then RAG-aware fine-tune
[Behaviour or format] Failing case -(wrong tone, schema, policy)-> Prompt with examples, structured outputs, then SFT
[Capability] Failing case -(cannot do the task)-> Stronger or reasoning model, then RFT
[Cost or latency] Correct but too slow or expensive -(same quality, less spend)-> Caching, smaller fine-tuned model, distillation
~~~

| Lever | What it fixes | What it does not fix | Running cost | Change cost |
| --- | --- | --- | --- | --- |
| Prompting | Behaviour, format, policy; cheap to iterate | Missing facts, missing skill | Input tokens per call | Minutes |
| Long context | Missing knowledge that fits; cross-document reasoning | Corpora over the window; per-user permissions | High per call, about 10x less with caching | Minutes |
| RAG | Large, changing, permissioned knowledge; citations | Behaviour; retrieval misses | Low per call plus an index | Days to weeks |
| Fine-tuning (SFT/LoRA) | Consistent format and tone, narrow skills, shorter prompts, small-model cost | New facts; tasks the base model cannot learn from a few thousand examples | Small model per call | Weeks; repeat when the base model retires |
| Preference or RL tuning | "Better versus worse" judgements, verifiable tasks | Knowledge | Same as SFT | Weeks plus a grader ([07.09](lesson:aieng:07.09)) |

The knowledge row is where most wrong choices happen. Ovadia et al. compared unsupervised fine-tuning with RAG on knowledge-intensive tasks and found RAG better for both facts the model had seen and entirely new ones. Gekhman et al. went further: in fine-tuning, examples that teach new facts are learned much more slowly than ones consistent with what the model knows, and once learned they increase hallucination roughly linearly. Fine-tuning teaches a model how to use what it knows; it is a poor way to teach it new things. [ML System Design 06.05](lesson:mlsd:06.05) adds the system-of-record rule: for exact values such as balances or prices, call a tool instead of retrieving text about them.

## Long context versus retrieval

If the whole knowledge base fits in the window, you can skip retrieval. Li et al. found that, given enough resources, long-context models beat RAG on average across their benchmarks, while RAG was far cheaper; their Self-Route sends each query to RAG first and falls back to long context when the model judges the retrieved chunks insufficient. Three things shift that trade in 2025–26:

- **Prompt caching** bills a repeated, byte-identical prefix at about a tenth of the input price on a hit, and cuts time to first token ([what your provider optimizes](lesson:aieng:06.05)). It only pays when the cache is warm: with a 5-minute TTL and a request every 40 minutes, every call is a write at 1.25× the input price.
- **Effective context is shorter than advertised.** Recall degrades for material in the middle of long prompts, and multi-hop questions over 100k tokens fail more often than needle tests suggest. Test at your real lengths.
- **Permissions and freshness.** One cached prefix per user, per tenant or per document version defeats the cache. Retrieval filters per request.

Long context fits a single large document per session (a contract, a codebase snapshot, a call transcript) and small stable corpora. RAG fits anything large, changing or access-controlled.

## What each option costs at three volumes

<div class="ex">
<div class="lab">Worked example</div>

**A support assistant over a 100k-token knowledge base.** Each request has 800 tokens of instructions, a 200-token question and a 300-token answer. RAG sends 4,000 tokens of retrieved chunks. Prices: a Sonnet-class model at \$3 per million input, \$15 per million output, \$0.30 cached read, \$3.75 cache write (Anthropic's list prices); a hosted fine-tuned small model at \$0.80 in and \$3.20 out, an H100 at \$2.50 an hour, and an index at \$70 a month (\$500 at the top volume). The last three are illustrative.

Per request: long context uncached \$0.3075, cached \$0.0353, RAG on the frontier model \$0.0195, RAG on the fine-tuned small model \$0.0044 (its prompt drops to 100 instruction tokens).

| Requests a month | Long context | Long context, cached | RAG, frontier | RAG, fine-tuned small (hosted) | RAG, fine-tuned 8B (self-hosted) |
| --- | --- | --- | --- | --- | --- |
| 1,000 | \$308 | \$383 (cache stays cold) | \$90 | \$74 | \$1,900 (one idle GPU) |
| 100,000 | \$30,750 | \$3,534 | \$2,020 | \$510 | \$1,900 |
| 10,000,000 | \$3,075,000 | \$353,400 | \$195,500 | \$44,500 | about \$7,800 (four GPUs) |

At 1,000 requests a month every option costs less than a day of engineering, so pick the one that is simplest to build and passes the eval, probably long context without caching. At 100,000, caching brings long context within 1.75× of RAG. If it answers better on your eval, keep it. A fine-tune saves about \$1,500 a month, which may not repay the weeks spent building data, evals and retraining. At 10 million, per-token cost dominates. The fine-tuned small model saves about \$150,000 a month against frontier RAG, and self-hosting saves another \$37,000 if you can run GPUs. In every column, the knowledge still arrives through retrieval or context; fine-tuning only changes which model reads it.

</div>

Add the hidden line items before you decide: labelling and eval work, a retraining run whenever the base model is deprecated, and the risk that a hosted fine-tuning service closes. As of November 2026, OpenAI's docs say its fine-tuning platform is closed to new users and existing customers can start training jobs only until 6 January 2027.

## Combined patterns that work

- **RAG plus a small fine-tune.** Train a small model on examples that include retrieved chunks and distractors, so it learns to answer from context and cite it. That is the RAFT recipe, and a common way to make a cheap model good at your RAG format. Labels usually come from a frontier teacher ([data for fine-tuning](lesson:aieng:07.08)).
- **Cached long context per session.** Load the document once, then answer follow-ups from cache.
- **Cascade.** A small fine-tuned model answers what it is confident about and escalates the rest to the frontier model ([cost budgeting](lesson:aieng:12.04)).
- **Prompt plus structured outputs** before any format fine-tune. Schema-constrained decoding fixes most format failures for free ([structured outputs](lesson:aieng:12.01)).

## Four scenarios

| Scenario | Dominant failure | Choice | Why not the others |
| --- | --- | --- | --- |
| Help-centre bot, 300k tokens of docs updated weekly, per-plan visibility | Missing knowledge | RAG with permission filters | Long context cannot filter per user without breaking the cache; fine-tuning bakes in stale docs |
| Contract review: one 80k-token contract, about 20 questions per session | Cross-references missed by chunked retrieval | Long context with caching: one \$0.30 write plus 19 reads at \$0.024 each, about \$0.76 of input per session versus \$4.80 uncached | RAG splits clauses that refer to each other; nothing to fine-tune |
| Invoice field extraction, 5M documents a month, fixed schema | Cost and latency (the frontier model is accurate) | Structured outputs now; then distil into a fine-tuned small model and gate on per-field accuracy | Long context and RAG solve nothing here |
| Marketing copy in a strict brand voice, using product facts | Behaviour, plus some knowledge | Prompt with approved examples plus RAG for facts; a small LoRA on about 500 approved pieces if voice still drifts | Fine-tuning facts in invents product claims |

## Where this breaks

- **Fine-tuned models expire with their base.** Keep the dataset, training config and eval so you can retrain on the successor.
- **Fine-tuning erodes safety and general skills.** Qi et al. showed that even benign fine-tuning data can weaken refusal behaviour. Re-run safety and general evals after training ([LoRA and QLoRA in practice](lesson:aieng:07.07)).
- **RAG debt.** Retrieval needs ingestion, freshness and permission plumbing that a long-context prototype never had ([keeping a RAG index correct](lesson:aieng:11.06)).
- **Comparing levers on different evals.** Run every option on the same labelled set, split by slice, or the cost table means nothing.

## What interviewers probe

- Why not fine-tune the documentation into the model? (Knowledge versus behaviour, the hallucination evidence, staleness.)
- How prompt caching changes the long-context cost argument, and when the cache stays cold.
- Cost per request and per month at a given volume, and where the crossover to a fine-tuned or self-hosted model sits.
- What you would do when retrieval finds the right chunk but the answer ignores it.

<div class="take">
<div class="lab">Key takeaway</div>

Classify failures first. Put knowledge in context (retrieval, or long context when it fits and caching keeps it cheap), and fix behaviour with prompts before weights. Fine-tune when volume makes a small model's savings worth the data, eval and retraining work.

</div>
