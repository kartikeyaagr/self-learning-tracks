---
id: "02.10"
course: "ai-engineering-course"
module: "02"
slug: "choose-and-fine-tune-embedding-model"
title: "Choosing and fine-tuning an embedding model"
keyIdeas: ["MTEB averages many task types; for RAG only the retrieval scores on data like yours matter, and leaderboard gaps of a point or two are noise.", "Shortlist on language, domain, input length, dimensions, prefixes and licence, then decide with recall@k on a few hundred of your own labelled queries.", "Contrastive fine-tuning pulls each query toward its positive and away from negatives; in-batch negatives are mostly easy, so mined hard negatives carry most of the signal.", "Many mined hard negatives are unlabelled positives. Filter them with a score margin against the positive, a rank window or a cross-encoder.", "Fine-tuning a small model beats a bigger general one when the domain has its own vocabulary and you have thousands of pairs; it also forces a full re-embed."]
question: "Your RAG system for internal engineering docs uses a top-ranked MTEB model, but recall@10 on your own queries is 0.62. Walk through how you would decide between switching models and fine-tuning, and how you would build the fine-tuning data."
answer: "First trust the 0.62 only if the eval is sound: a few hundred real queries with labelled relevant chunks, run with the model's required query and document prefixes. Then test two or three alternative models on the same queries; a paired comparison tells me whether any gap is real. If none clears the bar and the misses are domain terms (service names, error codes, internal acronyms), fine-tune. Training data: mine (query, positive) pairs from search logs, tickets that link to docs, or LLM-generated questions per chunk filtered for quality. For each query, retrieve the top 50 to 100 with the current model, drop the labelled positives, and keep hard negatives only if their score is below about 95% of the positive's score (or a cross-encoder says they are not relevant), to avoid training on false negatives. Train with an in-batch contrastive loss plus those negatives, evaluate on held-out queries from different documents, then re-embed the corpus."
rubric: ["Validates the eval itself (real queries, labels, prefixes) before acting", "Compares candidate models on the same queries before deciding to fine-tune", "Describes sources of training pairs and their risks", "Explains hard-negative mining and why false negatives must be filtered, with a concrete rule", "Uses a held-out split without document leakage and notes the re-embedding cost"]
sources: [{"title": "Enevoldsen et al.: MMTEB, Massive Multilingual Text Embedding Benchmark (2025)", "url": "https://arxiv.org/abs/2502.13595", "note": "The expanded benchmark behind the current MTEB leaderboard: 500+ tasks, 250+ languages."}, {"title": "Moreira et al.: NV-Retriever, effective hard-negative mining (2024)", "url": "https://arxiv.org/abs/2407.15831", "note": "Positive-aware mining: use the positive's score to remove false negatives."}, {"title": "Qu et al.: RocketQA (2020)", "url": "https://arxiv.org/abs/2010.08191", "note": "Denoised hard negatives with a cross-encoder, and cross-batch negatives."}, {"title": "Sentence Transformers: training overview", "url": "https://sbert.net/docs/sentence_transformer/training_overview.html", "note": "Datasets, losses and the trainer for fine-tuning embedding models."}]
practice: ["hard-negative-mining", "ann-recall-at-k", "precision-recall-at-k", "info-nce-loss", "clip-loss-hard-negatives"]
tools: ["MTEB leaderboard", "Sentence Transformers", "FAISS"]
---

## What MTEB measures

[From data to vectors](lesson:aieng:02.01) lists current embedding models by size, dimensions and context; this lesson is about picking among them and making one better on your data. The usual starting point is the MTEB leaderboard. The original MTEB (2022) covered 8 task types, 58 datasets and 112 languages; MMTEB (2025) grew it to more than 500 tasks in over 250 languages, with code retrieval, long documents and instruction following.

Those task types include classification, clustering, pair classification, reranking, retrieval, semantic textual similarity and summarisation. A RAG system cares about one: retrieval. The headline average mixes all of them, so a model can rank high on clustering and STS and sit lower on retrieval.

## Why the leaderboard misleads

- **The average is the wrong metric.** Sort by the retrieval column, and by the language and domain subsets closest to your data.
- **Training on the test.** Many models train on the training splits of datasets that also appear in the benchmark. The leaderboard flags how zero-shot a model is; a model that saw in-domain training data scores higher than it will on your corpus.
- **Small gaps are noise.** The top models are often within a point or two of each other on a mean over dozens of datasets. That difference rarely survives the change to your documents.
- **Your domain is not there.** Benchmark retrieval sets are mostly web, Wikipedia, science and question-answering. Internal runbooks, legal clauses, part numbers and ticket text behave differently.
- **Evaluation settings differ from yours.** Scores are computed with the model's own prefixes or instructions, its maximum input length, full-precision vectors and exact search. Truncated chunks, missing prefixes, quantized vectors ([Compression and quantization](lesson:aieng:02.05)) and an ANN index each cost some recall.

Size is also a weak guide. The MMTEB paper found the best publicly available model on its multilingual benchmark was multilingual-e5-large-instruct, at 560 million parameters, ahead of billion-parameter LLM-based embedders.

## Shortlist, then measure

| Question | Why it matters |
| --- | --- |
| Which languages, and mixed-language queries? | Multilingual models differ most here |
| How long are your chunks? | Text past the model's limit is silently cut |
| What dimensions and storage budget? | Matryoshka models let you truncate vectors |
| Does it need query and document prefixes or an instruction? | Asymmetric models lose recall without them ([Vector embeddings and semantic space](lesson:aieng:02.02)) |
| API or self-hosted, and under what licence? | Data terms, latency, cost per million tokens |

Then run the shortlist on your own labelled queries. Collect 200 to 500 queries from search logs, support tickets or subject-matter experts, and label the chunks that answer each one. LLM-generated questions per chunk are a fine supplement, but they tend to reuse the chunk's own words, which flatters lexical overlap. Measure recall@k at the k your pipeline actually passes on (the reranker's input depth, or the number of chunks in the prompt) plus MRR or nDCG; [Evaluating RAG](lesson:aieng:10.04) and [Retrieval and ranking metrics](lesson:mlsd:05.02) define them.

Mind the noise. With 300 queries and recall@10 of 0.80, the standard error is $\sqrt{0.8 \times 0.2 / 300} \approx 0.023$, so the 95% interval is about ±4.5 points. Two models two points apart are not distinguishable on separate runs; compare them on the same queries and count which queries flipped.

## Contrastive fine-tuning

Embedding models are trained to score a query $q$ above its positive passage $d^+$ and below negatives. With cosine similarity $s$ and temperature $\tau$, the loss per query is

$$
\mathcal{L} = -\log \frac{e^{s(q, d^+)/\tau}}{e^{s(q, d^+)/\tau} + \sum_{j} e^{s(q, d_j^-)/\tau}}
$$

With **in-batch negatives**, the other queries' positives in the batch serve as negatives for free, so a batch of 128 pairs gives each query 127 negatives (the InfoNCE practice problem). Sentence Transformers calls this `MultipleNegativesRankingLoss`. Larger batches mean more negatives and usually better models.

Most in-batch negatives are easy: a question about VPN setup against a passage about parental leave. The model learns little from them. **Hard negatives** are passages that look relevant but are not: the right topic, the wrong answer. DPR mined them with BM25; current practice mines them with an embedding model, often the one you are about to fine-tune.

~~~dg-flow Building fine-tuning data with mined hard negatives.
(query, positive) pairs -(embed + search top 100)-> Candidates -(drop labelled positives)-> Ranked negatives -(margin or cross-encoder filter)-> *Hard negatives -(add to batch)-> Contrastive loss
~~~

## False negatives

The catch is that the top-ranked "negatives" are often relevant passages nobody labelled. Training pushes the model away from correct answers. RocketQA tackled this by scoring mined negatives with a cross-encoder and discarding the ones it judged relevant. NV-Retriever proposed positive-aware filtering: use the positive's own score as an anchor and drop any candidate that scores too close to it. Its best setting kept negatives scoring below 95% of the positive's score.

<div class="ex">
<div class="lab">Worked example</div>

**mining negatives for one query**

Query: "cancel an order after it has shipped". Labelled positive: the returns policy chunk, cosine 0.82 with the current model. Threshold at 95% of the positive: $0.95 \times 0.82 = 0.779$.

| Rank | Candidate chunk | Score | Decision |
| --- | --- | --- | --- |
| 1 | "Stopping a shipment in transit" | 0.86 | Drop: above the positive, almost certainly relevant |
| 2 | "Returns policy (EU version)" | 0.80 | Drop: above 0.779, likely a duplicate positive |
| 3 | "Cancel an order before dispatch" | 0.77 | Keep: hard negative |
| 4 | "Change the delivery address" | 0.71 | Keep: hard negative |
| 5 | "Order confirmation emails" | 0.64 | Keep: hard negative |
| 6 | "Gift card balance" | 0.52 | Easy; not needed |

With three negatives per query, the kept set is ranks 3 to 5. Rank 3 is exactly what the model must learn to separate: same words, opposite condition. Ranks 1 and 2 would have taught it to push away two answers a user would accept.

</div>

Sentence Transformers implements the same filters in `mine_hard_negatives`:

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.util import mine_hard_negatives

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
train = mine_hard_negatives(
    pairs,                   # Dataset with (anchor, positive) columns
    model,
    num_negatives=3,
    range_min=0, range_max=100,
    relative_margin=0.05,    # keep a negative only if its score is at most 0.95 x the positive's
    sampling_strategy="top",
    output_format="n-tuple",
)
loss = losses.MultipleNegativesRankingLoss(model)   # in-batch + mined negatives
SentenceTransformerTrainer(model=model, train_dataset=train, loss=loss).train()
```

Other options there: skip the top few ranks (`range_min`), set an absolute score ceiling (`max_score`), or pass a cross-encoder so filtering uses a stronger relevance judge than the model being trained.

## When fine-tuning beats a bigger model

Fine-tune when the misses come from domain vocabulary a general model has not seen (internal service names, error codes, product SKUs, legal or clinical terms), you have a few thousand pairs or can generate and filter them, and a small fine-tuned model would meet a latency or cost target a large API model cannot. Gains of several recall points on in-domain queries are common in that setting; measure yours on held-out queries.

Prefer switching models when the problem is breadth (many languages, open-domain questions), when you have only a few hundred pairs, or when misses are really chunking or lexical problems: exact identifiers are often better served by adding BM25 ([Hybrid search and score fusion](lesson:aieng:11.02)) than by any embedding model.

Either way, a new or fine-tuned model lives in a new vector space. Every document must be re-embedded and the index rebuilt, with a dual-write migration so search keeps working ([Keeping a RAG index correct](lesson:aieng:11.06)).

## Where this breaks

- **Leakage in the eval.** Queries about the same document in both train and test inflate scores. Split by document, not by query.
- **Mining with the model you evaluate, then reporting it as fair.** Keep a frozen eval set that played no part in mining.
- **Negatives that are too hard.** With no false-negative filter, or a margin of zero, the model learns to separate near-duplicates and recall drops.
- **Forgetting general ability.** A model fine-tuned on narrow data can get worse on queries outside it. Keep a slice of general queries in the eval.
- **Dropping the prefixes during training or serving.** If the base model expects "query: " and "passage: ", use them in mining, training and inference alike.

## What interviewers probe

- Why the MTEB average is the wrong number for a RAG decision, and what you would look at instead.
- How you would build an embedding eval from scratch and how many queries you need.
- The contrastive loss with in-batch negatives, and why batch size matters.
- Hard-negative mining, false negatives, and a concrete filtering rule.
- When you would fine-tune versus switch models versus add lexical search, and what a fine-tune costs operationally.

<div class="take">
<div class="lab">Key takeaway</div>

pick embedding models with recall@k on your own labelled queries, not the MTEB average. Fine-tune with a contrastive loss when domain vocabulary is the problem, and spend your care on hard negatives: mine them with a retriever, then filter out the false negatives before they teach the model to push away right answers.

</div>
