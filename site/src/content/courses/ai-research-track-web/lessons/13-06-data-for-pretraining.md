---
id: "13.06"
course: "ai-research-track-web"
module: "13"
slug: "data-for-pretraining"
title: "Data for pretraining"
keyIdeas: ["A web pretraining pipeline is a funnel: extract text from crawled HTML, identify the language, apply heuristic quality filters, deduplicate, apply model-based quality filters, and remove unsafe or personal data. Most raw tokens never reach training.", "Heuristic filters (length, symbol and stop-word ratios, repeated lines, punctuation) remove obvious junk cheaply; model-based filters, such as a classifier trained to spot educational text, select what to keep and can shift benchmark results substantially.", "Exact deduplication hashes normalised documents or long substrings; near-duplicate detection uses MinHash over shingles with LSH bands, catching pairs above a Jaccard threshold of about $(1/b)^{1/r}$.", "The data mixture (web, code, books, maths, multilingual) matters as much as the volume. Temperature sampling $p_i \\propto n_i^{1/T}$ upweights small sources, but you must cap how many epochs each source is repeated.", "The tokenizer is part of the data pipeline: its compression (bytes per token) sets how much text a token budget buys, and choices such as byte fallback and digit splitting affect code and arithmetic."]
question: "Your near-duplicate step uses 100 MinHash values as 20 bands of 5 rows, and reviewers complain that it removes many documents that are merely on the same topic. You want to treat pairs with Jaccard similarity of about 0.8 and above as duplicates. Compute the candidate probability at Jaccard 0.5, 0.7, 0.8 and 0.9 for 20 x 5 and for 10 x 10, and recommend a configuration. What else would you check before rerunning the whole corpus?"
answer: "For 20 bands of 5 rows, $P = 1 - (1 - s^5)^{20}$: 0.47 at $s = 0.5$, 0.975 at 0.7, and essentially 1 at 0.8 and 0.9. The threshold $(1/20)^{1/5} \\approx 0.55$ is far too low, which explains the topical false positives: nearly half of pairs at Jaccard 0.5 are flagged. For 10 bands of 10 rows, $P = 1 - (1 - s^{10})^{10}$: 0.01 at 0.5, 0.25 at 0.7, 0.68 at 0.8 and 0.99 at 0.9, with threshold $(1/10)^{1/10} \\approx 0.79$. That matches the target, at the cost of missing about a third of pairs right at 0.8; more hashes (for example 14 bands of 8 rows, or more bands at 10 rows) sharpen the curve if recall near the threshold matters. Before rerunning: verify on a labelled sample of candidate pairs by computing their true Jaccard similarity; confirm the shingle size (word 5-grams behave very differently from character 5-grams); decide whether to deduplicate per snapshot or globally (FineWeb found per-snapshot better); keep a stable order so the same document survives each run; and run a small-model ablation on deduplicated versus original data, since the end goal is better models, not fewer documents."
sources: [{"title": "Penedo et al.: The FineWeb Datasets (2024)", "url": "https://arxiv.org/abs/2406.17557", "note": "The full pipeline with an ablation per step: extraction, filters, MinHash settings, per-snapshot deduplication and FineWeb-Edu."}, {"title": "Lee et al.: Deduplicating Training Data Makes Language Models Better (2021)", "url": "https://arxiv.org/abs/2107.06499", "note": "Exact substring deduplication with suffix arrays, near-duplicate MinHash, and the effects on memorisation."}, {"title": "Rae et al.: Scaling Language Models: Methods, Analysis and Insights from Training Gopher (2021)", "url": "https://arxiv.org/abs/2112.11446", "note": "The appendix on MassiveText: the quality filters and repetition rules that many later pipelines reuse."}]
practice: ["text-deduplication-minhash", "perform-stable-exact-deduplication", "temperature-based-data-mixtures", "extrapolate-data-mixture-scaling-curves", "train-deterministic-bpe-vocabulary"]
tags: ["pretraining data", "deduplication", "data mixtures"]
tools: ["datatrove", "Python", "fastText"]
---

## The funnel

Pretraining data quality is now as big a lever as model size. For the same compute budget, a better-filtered corpus can match a model trained on several times more unfiltered tokens. Most open recipes for web text look like this:

~~~dg-flow A web-text pipeline; each stage removes a large share of what reaches it.
Crawl snapshots (WARC) -> Text extraction from HTML -> Language ID -> Heuristic filters -> *Deduplication -> Model-based quality filter -> PII and safety filters -> Tokenise and shard
~~~

Text extraction matters more than it looks: extracting from raw HTML with a tool such as trafilatura gives cleaner text than relying on the crawl's own text dumps, and FineWeb's ablations showed the difference in downstream scores. FineWeb, built from 96 Common Crawl snapshots, yields about 15T tokens after the full pipeline, and its educational subset, FineWeb-Edu, is much smaller and scores markedly better on knowledge and reasoning benchmarks for the same training tokens.

## Heuristic filters

Rules inherited from C4, Gopher's MassiveText and later work are cheap to run on billions of documents:

- document length within a range, and mean word length within a range;
- the fraction of lines ending in terminal punctuation, and of words that are common stop words (real prose contains "the", "and", "of");
- symbol-to-word ratios (too many `#` or ellipses suggests lists, spam or code fragments);
- repeated lines, paragraphs and n-grams (boilerplate, navigation menus, cookie banners);
- blocklists of URLs or domains.

Each threshold is a hypothesis. The reliable way to choose them is ablation: train small models (around a billion parameters on tens of billions of tokens) on filtered versus unfiltered data, and compare on benchmarks that are sensitive at that scale. A filter that looks sensible can hurt, for example by removing most code or non-English text.

## Model-based filtering

A classifier can encode "what we want more of" directly. Older approaches trained a linear classifier to separate reference text (Wikipedia, books) from random web pages. FineWeb-Edu instead had a large instruction-tuned model rate a sample of pages for educational value, trained a small classifier on those ratings and kept pages above a threshold. Related methods score text by the perplexity of a reference language model. These filters are powerful and biased: they keep what resembles the reference, which can shrink diversity, so pipelines often mix the high-quality slice with a larger, less filtered pool.

## Deduplication

Duplicates waste compute, encourage memorisation and inflate benchmark scores when test items are duplicated in training data. Lee et al. found that standard corpora contained heavily repeated sequences and that deduplicating them reduced memorised output without hurting perplexity.

Exact deduplication hashes each document after normalisation (lower-casing, whitespace collapsing) and keeps the first occurrence in a stable order. Substring-level exact deduplication with suffix arrays removes repeated spans, such as licence text, inside otherwise different documents.

Near-duplicate detection uses MinHash. Represent each document as a set of shingles (word 5-grams, say). For a random hash function, the probability that two sets have the same minimum hash value equals their Jaccard similarity $s$. Compute $k = b \times r$ minimum hashes, group them into $b$ bands of $r$ rows, and make two documents candidates if any band matches entirely:

$$P(\text{candidate}) = 1 - (1 - s^r)^b.$$

That is an S-curve whose threshold sits near $(1/b)^{1/r}$. More rows per band sharpen it and raise it; more bands lower it. Candidates are then clustered (connected components) and one document per cluster is kept. FineWeb found that deduplicating each crawl snapshot separately worked better than deduplicating across all snapshots, because in older snapshots global deduplication removed most documents, and the small remainder it kept was of lower quality than what it removed.

## Mixtures

A pretraining corpus is a weighted mixture of sources. Sampling in proportion to size lets web text drown everything else; temperature sampling flattens it:

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}, \quad T \ge 1.$$

$T = 1$ is proportional sampling, and larger $T$ moves toward uniform. Every upweighted source is repeated: its epoch count is $p_i D / n_i$ for a run of $D$ tokens. A few epochs over high-quality data cost little, but beyond that returns diminish, so cap epochs and send the excess to the largest source. Mixture weights are chosen by small-scale ablations, by fitting per-domain scaling curves, or by optimisation methods such as training a proxy model. Many recipes also change the mixture late in training, annealing on a high-quality mix (maths, code, curated text) while the learning rate decays.

## Tokenizer choices

Tokenizers were covered earlier in the course; at pretraining scale three choices matter. Byte-level BPE with byte fallback can encode any input, so nothing is lost or mapped to unknown. Vocabulary size trades embedding and output-layer parameters against compression: larger vocabularies give fewer tokens per byte, so a fixed token budget covers more text, especially in multilingual and code data. Pre-tokenisation rules (splitting digits individually, keeping whitespace runs for code) change arithmetic and code performance. Always quote data budgets with the tokenizer, and compare models by bits per byte rather than per-token loss when tokenizers differ.

<div class="ex">
<div class="lab">Worked example</div>

**choosing bands and a mixture**

MinHash with $b = 14$ bands of $r = 8$: threshold about $(1/14)^{1/8} = 0.72$. Pairs at Jaccard 0.5 become candidates with probability $1 - (1 - 0.5^8)^{14} = 0.053$, at 0.75 with 0.772, at 0.8 with 0.924 and at 0.9 above 0.999.

Mixture: sources of 900B web, 80B code, 15B books and 5B wiki tokens. At $T = 2$ the weights are proportional to $\sqrt{n_i}$: 66.6, 19.9, 8.6 and 5.0 percent. A 1T-token run then sees web 0.74 epochs, code 2.48, books 5.73 and wiki 9.93. Wiki and books are repeated too often; capping them near four epochs and moving the surplus to web is the usual fix.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

treat data as an engineered pipeline (extract, filter, deduplicate, mix) whose every rule is validated by small-model ablations, and report data budgets together with the tokenizer and the epoch count per source.

</div>
