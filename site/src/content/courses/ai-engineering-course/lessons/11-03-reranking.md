---
id: "11.03"
course: "ai-engineering-course"
module: "11"
slug: "reranking"
title: "Reranking"
keyIdeas: ["Retrieval is tuned for recall over millions of chunks; a reranker is tuned for precision over the top 50 to 200. It can only reorder what retrieval found.", "Bi-encoders embed query and document separately, cross-encoders read them together (most accurate, one model pass per pair), and late interaction (ColBERT) keeps a vector per token and scores with MaxSim.", "LLM rerankers score pointwise (a relevance probability per document) or listwise (reorder a window of documents); they are strong but cost many tokens and, listwise, several sequential calls.", "Rerank depth is a trade-off: candidate recall rises with depth while latency rises in batch-sized steps and cost rises linearly; pick depth on a recall-versus-latency table from your own data.", "After reranking, MMR or a per-document cap removes near-duplicate chunks so the prompt covers more than one source."]
question: "Your RAG pipeline retrieves 200 chunks and must answer in under 1.5 s with the reranker allowed about 150 ms. Compare a cross-encoder, ColBERT and an LLM listwise reranker for this slot, and explain how you would choose the rerank depth."
answer: "A cross-encoder reads each (query, chunk) pair jointly, so it is the most accurate per pair but costs one forward pass per candidate; on a GPU, batched, 100 to 200 short pairs can fit in about 100 ms, and hosted rerank APIs bill per pair or per token. ColBERT precomputes per-token document vectors and scores with MaxSim, so it is much cheaper at query time and can even serve as the first stage, but its index is several times larger than single-vector embeddings. An LLM listwise reranker (RankGPT-style sliding windows) is strong on hard queries but needs several sequential calls and tens of thousands of input tokens, so it does not fit 150 ms. Choose the cross-encoder. For depth, measure first-stage recall@N and final recall@5 after reranking for N = 20, 50, 100, 200, measure p95 reranker latency at each, and take the smallest N that is within a point of the best recall and inside the budget."
rubric: ["States that reranking cannot recover documents the first stage missed", "Explains cross-encoder versus bi-encoder (joint versus separate encoding) and its per-pair cost", "Describes ColBERT late interaction and its storage cost", "Explains why listwise LLM reranking misses a tight latency budget (sequential calls, tokens)", "Chooses depth from measured recall and p95 latency, not a default", "Mentions batching or cost per query"]
sources: [{"title": "Nogueira and Cho: Passage Re-ranking with BERT (2019)", "url": "https://arxiv.org/abs/1901.04085", "note": "The cross-encoder reranker over BM25 candidates on MS MARCO."}, {"title": "Santhanam et al.: ColBERTv2 (2021)", "url": "https://arxiv.org/abs/2112.01488", "note": "Late interaction with residual compression, cutting index size 6 to 10 times."}, {"title": "Sun et al.: Is ChatGPT Good at Search? LLMs as re-ranking agents (2023)", "url": "https://arxiv.org/abs/2304.09542", "note": "RankGPT: listwise permutation reranking with a sliding window."}, {"title": "Anthropic: Introducing Contextual Retrieval", "url": "https://www.anthropic.com/news/contextual-retrieval", "note": "Reranking the top 150 down to 20, measured end to end."}]
practice: ["rerank-depth-budget", "mmr-diversity-rerank", "ndcg-at-k", "hit-rate-and-mrr"]
tools: ["Cohere Rerank", "Voyage rerank", "bge-reranker-v2-m3", "Qwen3-Reranker", "ColBERTv2", "RankGPT"]
---

## Why a second stage

The first stage ([vector search](lesson:aieng:02.06), [BM25](lesson:aieng:11.01), [fused](lesson:aieng:11.02)) must look at millions of chunks in tens of milliseconds, so it uses representations computed before the query arrived. That makes it good at recall (the right chunk is somewhere in the top 100) and mediocre at precision (it is not reliably in the top 5). A reranker spends far more compute per candidate on a short list. [Search execution](lesson:aieng:02.08) shows where it sits on the query path; this lesson covers the model choices, the depth decision and the failure modes.

~~~dg-flow A funnel: each stage sees fewer items and spends more per item.
10M chunks -(BM25 + ANN, ~20 ms)-> 200 candidates -(*reranker, ~100 ms)-> Top 20 -(MMR, cap per doc)-> 5 chunks in the prompt
~~~

One rule shapes everything that follows: **a reranker can only reorder what it is given.** Final recall@5 can never exceed first-stage recall at the rerank depth. When answers are missing, measure first-stage recall@N before blaming the reranker.

## Three ways to score a pair

| | Bi-encoder | Cross-encoder | Late interaction (ColBERT) |
| --- | --- | --- | --- |
| Encoding | Query and document separately, one vector each | Query and document concatenated, read together | Separately, one vector per token |
| Score | Dot product or cosine | A relevance head on the joint encoding | $\sum_i \max_j q_i \cdot d_j$ (MaxSim) |
| Precomputed | All document vectors | Nothing | All document token vectors |
| Query-time cost | One query embedding plus ANN | One full forward pass per candidate | One query encoding plus cheap MaxSim per candidate |
| Accuracy | Lowest | Highest | Between, closer to cross-encoder |
| Index size | One vector per chunk | No index | One vector per token |

A **cross-encoder** (Nogueira and Cho put BERT over BM25 candidates on MS MARCO) lets every query token attend to every document token, so it sees that "not supported on Windows" answers "does it run on Windows?" in a way two independent vectors cannot. The price is one transformer pass per candidate: reranking 100 chunks of 300 tokens processes about 33,000 tokens per query, similar to a 33k-token prefill. Classic cross-encoders were trained with 512-token inputs and silently truncate longer chunks; newer ones accept far more.

**Late interaction** (ColBERT, Khattab and Zaharia) stores a small vector for every document token. For each query token it takes the best-matching document token and sums those maxima. Documents are encoded offline, so query time is cheap; the ColBERT paper reported about two orders of magnitude lower latency than BERT reranking at competitive quality. The cost moves to storage: 10M chunks of 300 tokens at 128 dimensions in fp16 is $10^7 \times 300 \times 128 \times 2$ bytes $\approx 768$ GB. ColBERTv2's residual compression (a centroid id plus a 1- or 2-bit residual per dimension) cuts footprint 6 to 10 times; at 2 bits that is roughly 36 bytes per token, about 108 GB here, against about 20 GB for one 1,024-dimensional fp16 vector per chunk. ColBERT works as a reranker over another retriever's candidates, or as the first stage with an engine such as PLAID.

## LLM rerankers

An instruction-tuned LLM can judge relevance too, in three styles:

- **Pointwise:** prompt "Does this passage answer the query? yes/no" and use the probability of "yes" as the score. Qwen3-Reranker (0.6B, 4B and 8B open weights) works this way; it is a cross-encoder built on an LLM.
- **Listwise:** give the model a numbered window of passages and ask for a permutation. RankGPT (Sun et al.) slides a window of 20 with step 10 from the bottom of the list to the top, so the best passages bubble up. For 100 candidates that is 9 sequential calls of roughly 6,000 input tokens each.
- **Pairwise:** ask which of two passages is better. The number of comparisons grows quadratically with the list (or as $n \log n$ with a sorting scheme), so it is rarely used beyond short lists.

Listwise reranking with a frontier model can beat small cross-encoders on hard, reasoning-heavy queries, but it costs seconds and tens of thousands of tokens per query, the permutation can come back malformed (missing or repeated ids), and results shift with the order passages are presented. The common production pattern is to use the LLM offline as a teacher, label pairs, and distil into a cross-encoder that serves online.

## Choosing a reranker

| Option | Examples (2025–26) | Latency and cost profile |
| --- | --- | --- |
| Hosted rerank API | Cohere Rerank, Voyage rerank, Jina | No GPUs to run; a network hop plus per-request or per-token billing; long context per document (Voyage documents a 32k-token limit for query plus document) |
| Open-weights cross-encoder | bge-reranker-v2-m3, mxbai-rerank, Qwen3-Reranker-0.6B | You host on GPU; tens of ms per batch; cost is GPU time; fine-tunable on your data |
| Larger LLM-based pointwise | Qwen3-Reranker-4B or 8B | Better on hard queries, several times the compute |
| Late interaction | ColBERTv2, newer small ColBERT models via PyLate | Cheap queries, large index; can replace the first stage |
| Listwise LLM | RankGPT-style prompting of a frontier model | Seconds per query; offline labelling or low-volume, high-value queries |

Hosted rerankers bill on what you send. Voyage, for instance, counts query tokens times the number of documents plus the document tokens, so long queries multiply cost by depth. Model lists and prices change quickly; check the provider's page on the day you choose, and run the candidates on your own labelled queries, because public leaderboard gaps of a point rarely survive a domain shift.

## Rerank depth

<div class="ex">
<div class="lab">Worked example</div>

**picking N for a 120 ms reranker budget (illustrative numbers)**

Self-hosted cross-encoder: 10 ms fixed (tokenise, transfer) plus 25 ms per batch of 32 pairs. Measured on 400 labelled queries:

| Depth N | First-stage recall@N | Recall@5 after rerank | Batches | Latency | Tokens per query |
| --- | --- | --- | --- | --- | --- |
| 20 | 0.80 | 0.72 | 1 | 35 ms | 6,600 |
| 50 | 0.89 | 0.79 | 2 | 60 ms | 16,500 |
| 100 | 0.94 | 0.82 | 4 | 110 ms | 33,000 |
| 200 | 0.97 | 0.83 | 7 | 185 ms | 66,000 |

N = 200 is over budget. N = 100 fits with 10 ms to spare and loses one point of recall@5 against 200. Latency moves in batch steps: N = 64 costs the same 60 ms as 50 (two batches), N = 65 needs a third, so try depths that fill whole batches.

If the same reranker were a hosted API at an illustrative \$0.05 per million tokens, N = 100 with 330 tokens per pair costs $33{,}000 \times 0.05 / 10^6 = 0.00165$ dollars per query, or \$1,650 per million queries; N = 200 doubles that for one point.

</div>

Measure p95, not the mean: a reranker on a shared GPU queues under load, and the tail is what blows the end-to-end budget ([Search execution](lesson:aieng:02.08) has the full latency funnel). Anthropic's contextual retrieval setup is a useful reference point: it retrieved 150 chunks, reranked them and kept 20, and reranking took the top-20 failure rate from 2.9% to 1.9% on their benchmark.

## Diversity after reranking

Rerankers score each chunk independently, so three overlapping chunks from the same page can take the top three places and crowd out the second source the answer needed. Maximal marginal relevance picks greedily by $\lambda\,\text{rel}(i) - (1-\lambda)\max_{j \in S}\text{sim}(i, j)$; [retrieval and ranking pipelines](lesson:mlsd:06.02) gives the formula in context and the `mmr-diversity-rerank` problem implements it. A simpler cap of at most two chunks per document often does most of the work.

## Where this breaks

- **Low first-stage recall.** The reranker gets blamed for documents it never saw. Plot recall@N before tuning anything else.
- **Truncation.** A 2,000-token chunk fed to a 512-token cross-encoder is scored on its first quarter. Match chunk size to the reranker's window ([Chunking and document parsing](lesson:aieng:11.04)).
- **Missing context.** Rerank the chunk with its title and section heading attached; a bare paragraph is ambiguous to any model.
- **Thresholds.** Dropping chunks below a reranker score lets you send fewer than k when nothing is relevant, but raw cross-encoder logits are not probabilities and shift between model versions. Calibrate the threshold on labelled data and re-check it on every model upgrade (some hosted models now advertise calibrated scores).
- **Signals the reranker ignores.** Freshness, authority and permissions are not in the text. Apply ACL filters before reranking and blend recency after it.

## What interviewers probe

- "Why not use the cross-encoder for retrieval?" One forward pass per document per query: millions of passes.
- "Bi-encoder versus cross-encoder versus ColBERT?" Where interaction happens, what is precomputed, what it costs.
- "How deep do you rerank?" Show a recall@N curve and a batch-stepped latency model.
- "Your reranker made things worse." Check truncation, domain shift, missing titles, and whether first-stage recall changed.

<div class="take">
<div class="lab">Key takeaway</div>

Retrieve for recall, rerank for precision. Cross-encoders are the accurate default for a 50 to 200 candidate list, late interaction trades index size for cheap queries, and LLM rerankers are best used offline or as teachers. Pick the depth from your own recall-versus-p95-latency table, and remember the reranker can never surface a chunk the first stage missed.

</div>
