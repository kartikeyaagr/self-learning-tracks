---
id: "13.06"
course: "ai-research-track-web"
module: "13"
slug: "data-for-pretraining"
title: "Data for pretraining"
keyIdeas: ["A web pretraining pipeline is a funnel: extract text from crawled HTML, identify the language, apply heuristic quality filters, deduplicate, apply model-based quality filters, and remove unsafe or personal data. Most raw tokens never reach training.", "Heuristic filters (length, stop-word share, lines ending in punctuation, symbol ratio, repeated lines) remove obvious junk cheaply but also some good text; model-based filters, such as a classifier trained on LLM ratings of educational value, select what to keep and can shift benchmark results substantially.", "Exact deduplication hashes normalised documents in a stable order. Near-duplicates are found with MinHash, whose signatures agree at a rate equal to the Jaccard similarity of the shingle sets; banding into $b$ bands of $r$ rows flags pairs above a threshold of about $(1/b)^{1/r}$.", "The data mixture matters as much as the volume. Temperature sampling $p_i \\propto n_i^{1/T}$ upweights small sources, but each source is then repeated $p_i D / n_i$ times, so cap epochs (around 4) and move the surplus to the largest source.", "The tokenizer is part of the data pipeline: its bytes per token set how much text a token budget buys (a prose-only tokenizer compresses code poorly), and models with different tokenizers should be compared in bits per byte."]
question: "Your near-duplicate step uses 100 MinHash values as 20 bands of 5 rows, and reviewers complain that it removes many documents that are merely on the same topic. You want to treat pairs with Jaccard similarity of about 0.8 and above as duplicates. Compute the candidate probability at Jaccard 0.5, 0.7, 0.8 and 0.9 for 20 x 5 and for 10 x 10, and recommend a configuration. What else would you check before rerunning the whole corpus?"
answer: "For 20 bands of 5 rows, $P = 1 - (1 - s^5)^{20}$: 0.47 at $s = 0.5$, 0.975 at 0.7, and essentially 1 at 0.8 and 0.9. The threshold $(1/20)^{1/5} \\approx 0.55$ is far too low, which explains the topical false positives: nearly half of pairs at Jaccard 0.5 are flagged. For 10 bands of 10 rows, $P = 1 - (1 - s^{10})^{10}$: 0.01 at 0.5, 0.25 at 0.7, 0.68 at 0.8 and 0.99 at 0.9, with threshold $(1/10)^{1/10} \\approx 0.79$. That matches the target, at the cost of missing about a third of pairs right at 0.8; more hashes (for example 14 bands of 8 rows, or more bands at 10 rows) sharpen the curve if recall near the threshold matters. Before rerunning: verify on a labelled sample of candidate pairs by computing their true Jaccard similarity; confirm the shingle size (word 5-grams behave very differently from character 5-grams, and short shingles inflate similarity between same-topic pages); decide whether to deduplicate per snapshot or globally (FineWeb found per-snapshot better); keep a stable document order so the same copy survives each run; measure what share of each domain and language the new setting removes; and run a small-model ablation on deduplicated versus original data, since the end goal is better models, not fewer documents."
sources: [{"title": "Penedo et al.: The FineWeb Datasets (2024)", "url": "https://arxiv.org/abs/2406.17557", "note": "The full pipeline with an ablation per step: extraction, filters, MinHash settings, per-snapshot deduplication and FineWeb-Edu."}, {"title": "Lee et al.: Deduplicating Training Data Makes Language Models Better (2021)", "url": "https://arxiv.org/abs/2107.06499", "note": "Exact substring deduplication with suffix arrays, near-duplicate MinHash, and the effects on memorisation."}, {"title": "Rae et al.: Scaling Language Models: Methods, Analysis and Insights from Training Gopher (2021)", "url": "https://arxiv.org/abs/2112.11446", "note": "The appendix on MassiveText: the quality filters and repetition rules that many later pipelines reuse."}]
practice: ["text-deduplication-minhash", "perform-stable-exact-deduplication", "temperature-based-data-mixtures", "extrapolate-data-mixture-scaling-curves", "train-deterministic-bpe-vocabulary"]
tags: ["pretraining data", "deduplication", "data mixtures"]
tools: ["datatrove", "Python", "fastText"]
---

## What you will learn

- What each stage of a web-text pipeline removes, and why data is as big a lever as model size.
- How heuristic filters separate prose from junk, and what good text they wrongly drop.
- How exact deduplication works, and how MinHash with banding finds near-duplicates at scale.
- How to mix sources with temperature sampling without over-repeating small ones.
- Why the tokenizer belongs to the data pipeline.

*Before you start:* lessons 13.05 (scaling laws and the token budget $D$) and 13.01 ($C \approx 6ND$), plus what BPE tokenizers are. Sets, hashing and basic probability cover the maths.

## From token budgets to tokens

Lesson 13.05 assumed *fresh tokens of constant quality*. A Chinchilla-style 70B model wants about 1.4T of them, and for most open models they come from the public web, which is mostly menus, spam, duplicated pages and filler. This lesson turns a raw crawl into training tokens. For a fixed compute budget, a well-filtered corpus can match a model trained on several times more unfiltered tokens, so every rule in the pipeline is a modelling decision and gets tested like one.

## The idea in one picture

To see a pipeline end to end, we built a toy "crawl" of 2,000 documents and ran real filters over it:

- 1,200 genuine documents: English documentation text from the docstrings of installed Python libraries;
- 400 synthetic junk pages: menus with cookie banners, keyword spam, symbol soup, product tables, stubs like "Page not found.";
- 200 exact and 200 lightly edited copies (3% of words swapped) of genuine documents.

| Stage | Documents left | Removed | Mostly
| --- | --- | --- | --- |
| Raw toy crawl | 2,000 | 0 | starting point
| Length 50 to 100k words | 1,884 | 116 | stubs (80), short real docs (27)
| Repeated lines < 15% | 1,805 | 79 | menus and banners (71)
| Symbol-to-word ratio < 0.1 | 1,724 | 81 | symbol soup (80)
| Lines ending in punctuation > 12% | 1,493 | 231 | tables (80), spam (80), real docs (42)
| Stop-word ratio >= 0.06 | 1,488 | 5 | real docs (4)
| Exact dedup (hash) | 1,303 | 185 | copies
| MinHash near-dedup | 1,133 | 170 | edited copies

~~~dg-fig 13.06-funnel.svg
Our toy corpus shrinking stage by stage. Every junk page is gone after five cheap rules, deduplication removes another 355 documents, and 57% of the starting documents survive. Real web pipelines keep a much smaller share.
~~~

Three lessons. Cheap rules catch all 400 junk pages. They are not free: 80 of the 1,200 genuine documents (6.7%) went too. And duplicates survive every quality rule (a copy of good text *is* good text), so they need their own stage. A **heuristic filter** is a hand-written rule on simple document statistics; a **near-duplicate** differs from another document only slightly (a date, a footer, a few words).

The full pipeline that real recipes (C4, Gopher's MassiveText, RefinedWeb, FineWeb) roughly share is:

~~~dg-flow A web-text pipeline; each stage removes a large share of what reaches it.
Crawl snapshots (WARC) -> Text extraction from HTML -> Language ID -> Heuristic filters -> *Deduplication -> Model-based quality filter -> PII and safety filters -> Tokenise and shard
~~~

## Step 1: get text out of HTML

Common Crawl ships each snapshot as raw HTML (WARC files) and as pre-extracted text (WET files). Extraction decides what the "document" is: article text versus menus, sidebars and footers. Running a main-content extractor such as trafilatura on the raw HTML gives cleaner text than the WET dumps, and FineWeb's ablations showed the difference in benchmark scores. Next, **language identification**: a small fastText classifier predicts each document's language with a confidence, and you keep confident documents in the languages you want.

## Step 2: heuristic quality filters

For a document with $W$ words and $L$ non-empty lines, typical cheap features are:

$$\text{stop ratio} = \frac{\#\{\text{stop words}\}}{W}, \qquad \text{punct ratio} = \frac{\#\{\text{lines ending in . ! ? or a quote}\}}{L}, \qquad \text{dup-line ratio} = 1 - \frac{\#\{\text{distinct lines}\}}{L}.$$

**Stop words** are common function words ("the", "of", "and"): real sentences cannot avoid them, while spam, menus and tables contain almost none. The punctuation ratio catches tables and lists; the dup-line ratio catches repeated boilerplate.

~~~dg-fig 13.06-heuristics.svg
Prose and junk barely overlap on these two statistics. Real prose has about 30% stop words and most lines end in punctuation; junk sits near zero on both, so one threshold per feature (dashed) separates them. The left tail of the blue histograms is the real text these rules still remove.
~~~

In our toy corpus the median genuine document has a stop-word share of 0.30 and 86% of lines ending in punctuation; spam and tables score 0 on punctuation. Our rules follow Gopher (stated from memory: 50 to 100,000 words, mean word length 3 to 10, `#`-or-ellipsis-to-word ratio at most 0.1, a minimum of English stop words), C4 (keep lines ending in terminal punctuation, drop pages with "lorem ipsum" or curly braces) and FineWeb (a rule on the share of lines ending in punctuation).

Each threshold is a hypothesis, and hypotheses carry bias. Our punctuation rule removed 42 genuine documents whose paragraphs ended in lists. On real data the C4 curly-brace rule removes most code, and English stop-word rules wipe out other languages. Choose thresholds by ablation (see "Checking" below).

## Step 3: exact deduplication

The web is full of mirrors, syndicated news and templated pages. Duplicates waste compute, encourage memorisation and inflate benchmark scores when test items leak into training. Lee et al. showed that deduplicating standard corpora reduced memorised output without hurting perplexity.

Exact deduplication is simple:

1. **Normalise** each document: lower-case it and collapse runs of whitespace, so trivial differences do not hide a copy.
2. **Hash** the normalised text (SHA-1 or SHA-256), turning each document into a short fingerprint.
3. Walk the corpus in a **stable order** and keep the first document with each fingerprint.

With an unstable order a different copy survives each run and results stop being reproducible. A finer version removes repeated *substrings*: Lee et al. used a suffix array to find long spans (licence blocks, disclaimers) repeated across otherwise different documents and cut them out.

## Step 4: near-duplicates with MinHash

Exact hashing misses a page that differs by one word, and comparing all pairs of $10^9$ documents ($5 \times 10^{17}$ pairs) is impossible.

**Similarity.** Represent each document as its set of **shingles**: all runs of $n$ consecutive words (word 5-grams are common). Two documents are compared with the **Jaccard similarity**

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|},$$

shared shingles over distinct shingles in either: 1 for identical documents, near 0 for unrelated ones.

**MinHash.** Apply a random hash function $h$ to every shingle and keep only the smallest value, $\min_{x \in A} h(x)$. The key fact is

$$P\Big(\min_{x \in A} h(x) = \min_{x \in B} h(x)\Big) = J(A, B).$$

Why: the shingle of $A \cup B$ with the smallest hash is equally likely to be any of them, and the two minimums agree exactly when it lies in both sets. With $k$ hash functions a document's **signature** is its $k$ minimums, and the fraction of matching positions estimates $J$.

<div class="ex">
<div class="lab">Worked example</div>

**Jaccard and MinHash by hand**

Document A is "the cat sat on the mat today" and B is "the cat sat on the red mat today". Use word 3-grams.

A's shingles: "the cat sat", "cat sat on", "sat on the", "on the mat", "the mat today" (5 shingles).

B's shingles: "the cat sat", "cat sat on", "sat on the", "on the red", "the red mat", "red mat today" (6 shingles).

Shared: the first three, so $|A \cap B| = 3$. Distinct in either: $5 + 6 - 3 = 8$. So $J = 3/8 = 0.375$.

MinHash view: the smallest-hash shingle of the union is one of the 3 shared ones with probability $3/8$, so each signature position agrees with probability 0.375. The code below estimates 0.352 from 128 hashes.

Note how one added word destroyed 3 of A's 5 shingles: on short documents Jaccard is very sensitive to single edits.

</div>

On 190 document pairs from our toy corpus, 112 hashes estimated Jaccard with an error standard deviation of 0.037 and almost no bias (left panel below).

**Banding (LSH).** Comparing all pairs of signatures is still quadratic. **Locality-sensitive hashing** splits the $k = b \times r$ values into $b$ bands of $r$ rows and hashes each band into a bucket. Two documents are a **candidate pair** if any band lands in the same bucket, which needs all $r$ rows to agree. A band matches with probability $s^r$ ($s$ is the pair's Jaccard), so

$$P(\text{candidate}) = 1 - (1 - s^r)^b.$$

This S-shaped curve has its steep part near $s^* \approx (1/b)^{1/r}$.

~~~dg-fig 13.06-minhash.svg
Left: MinHash signature agreement tracks the true Jaccard similarity on 190 document pairs. Right: banding turns that estimate into a soft threshold. More rows per band move the step right and sharpen it; more bands move it left.
~~~

Number check with $b = 14$, $r = 8$ (112 hashes): $s^* = (1/14)^{1/8} = 0.72$. A pair at $s = 0.5$ becomes a candidate with probability $1 - (1 - 0.5^8)^{14} = 0.053$; at $s = 0.8$ it is 0.924; at $s = 0.9$ it is 0.9996. Our pair from the example ($s = 0.375$) is flagged with probability 0.0055.

Candidate pairs become edges; union-find groups them into clusters, and one document per cluster is kept.

## Step 5: model-based quality filtering

Heuristics remove what is obviously bad; a classifier can select what is actively good:

- **Reference classifiers**: a fastText-style classifier separates trusted text (Wikipedia, books) from random web pages; keep pages that look like the reference.
- **Perplexity filters**: keep pages that a small reference language model finds unsurprising.
- **LLM-annotated classifiers**: FineWeb-Edu asked a large instruction-tuned model (Llama 3 70B, from memory) to rate a few hundred thousand pages for educational value from 0 to 5, trained a small classifier on the ratings, and kept pages scoring 3 or more. The result (about 1.3T tokens, from memory, versus about 15T for FineWeb from 96 crawl snapshots) scores markedly better on knowledge and reasoning benchmarks for the same training tokens.

All of these keep what resembles the reference, which shrinks diversity, so recipes often mix the high-quality slice with a larger, less filtered pool. Personal data and unsafe content are removed in a final pass with regular expressions, blocklists and classifiers.

## Step 6: mixing sources

A corpus is a weighted **mixture** of sources (web, code, books, maths, other languages). Sampling in proportion to size lets web drown everything else; **temperature sampling** flattens the mix:

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}, \qquad T \ge 1,$$

where $n_i$ is the tokens available in source $i$, $p_i$ its share of training tokens and $T$ the temperature ($T = 1$ is proportional; large $T$ approaches uniform). The price: in a run of $D$ tokens, source $i$ is seen

$$\text{epochs}_i = \frac{p_i D}{n_i}$$

times. Muennighoff et al. (stated from memory) found that up to about 4 epochs of repeated data behave almost like fresh data, with returns falling quickly after that. So cap epochs per source and send the surplus to the largest source.

<div class="ex">
<div class="lab">Worked example</div>

**a four-source mixture for a 1T-token run**

Sources: web 900B, code 80B, books 15B, wiki 5B tokens. Use $T = 2$, so weights are proportional to $\sqrt{n_i}$.

Square roots: $\sqrt{900} = 30$, $\sqrt{80} = 8.944$, $\sqrt{15} = 3.873$, $\sqrt{5} = 2.236$. Their sum is 45.053.

Shares: $30/45.053 = 66.6\%$, $8.944/45.053 = 19.9\%$, $3.873/45.053 = 8.6\%$, $2.236/45.053 = 5.0\%$.

Tokens drawn from a 1T run: 665.9B, 198.5B, 86.0B and 49.6B. Epochs: $665.9/900 = 0.74$, $198.5/80 = 2.48$, $86.0/15 = 5.73$, $49.6/5 = 9.93$.

Wiki would be seen almost ten times. Cap books and wiki at 4 epochs: they supply 60B and 20B. The surplus is $(86.0 - 60) + (49.6 - 20) = 55.6$B, which goes to web: $665.9 + 55.6 = 721.5$B, or 0.80 epochs.

</div>

~~~dg-fig 13.06-mixture.svg
Left: raising the temperature moves tokens from web toward the small sources. Right: the cost of that, measured in epochs. At T = 2 books and wiki pass the 4-epoch line; capping them shifts about 56B tokens back to web.
~~~

Weights are chosen by small-model ablations, by extrapolating per-domain scaling curves, or by proxy-model optimisation (DoReMi is one). Many recipes also **anneal**: switch to a high-quality mix (maths, code, curated text) while the learning rate decays.

## Step 7: the tokenizer is part of the data

A token is whatever the tokenizer says it is, so its **compression** (bytes per token) decides how much text a token budget buys. We trained two small byte-level BPE tokenizers (256 byte tokens plus merges, up to 4,096): one on documentation prose, one on a prose and Python code mix, and measured held-out bytes per token.

~~~dg-fig 13.06-tokenizer.svg
Larger vocabularies compress better. The prose-only tokenizer (blue) reaches 3.49 bytes per token on prose but only 1.96 on code; adding code to its training data (orange) lifts code to 3.37 while costing prose only a little (3.36).
~~~

The prose-only tokenizer needs about 1.7 times as many tokens for the same code ($3.37 / 1.96$). Three choices matter at scale:

- **Byte fallback**: byte-level BPE can encode any input, so nothing becomes an unknown token.
- **Vocabulary size**: bigger vocabularies compress better but add embedding and output parameters.
- **Pre-tokenisation rules**: splitting digits one by one helps arithmetic; keeping runs of spaces as tokens helps code.

Compare models with different tokenizers in **bits per byte**: $\text{bpb} = \dfrac{\text{loss in nats per token}}{\ln 2 \times \text{bytes per token}}$. A loss of 2.0 nats per token at 4 bytes per token is $2.0 / (0.693 \times 4) = 0.72$ bits per byte.

## The pipeline in code

A minimal runnable version of exact dedup, MinHash and banding. Production tools (datatrove, NeMo Curator) do the same in parallel over shards.

```python
import hashlib, re
import numpy as np

def normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def shingles(text, n=3):
    w = normalise(text).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b)

P61 = (1 << 61) - 1                      # a large prime for universal hashing
def minhash(sh, k=128, seed=0):
    rng = np.random.default_rng(seed)    # same seed => same k hash functions for every doc
    a = rng.integers(1, P61, k, dtype=np.uint64)
    b = rng.integers(0, P61, k, dtype=np.uint64)
    x = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")
                  for s in sh], dtype=np.uint64)
    return ((x[:, None] * a + b) & np.uint64(P61)).min(axis=0)   # k minimum values

docs = ["The cat sat on the mat today",
        "the  cat sat on the mat today",          # exact duplicate after normalising
        "The cat sat on the red mat today",       # near duplicate
        "Stock markets fell sharply on Monday"]   # unrelated

# 1. exact dedup: hash the normalised text, keep the first occurrence
seen, kept = set(), []
for i, d in enumerate(docs):
    h = hashlib.sha256(normalise(d).encode()).hexdigest()
    if h not in seen:
        seen.add(h); kept.append(i)
print(kept)                                            # [0, 2, 3]

# 2. near dedup: true Jaccard vs MinHash estimate
A, B = shingles(docs[0]), shingles(docs[2])
print(len(A), len(B), round(jaccard(A, B), 3))         # 5 6 0.375
sa, sb = minhash(A), minhash(B)
print(round(float(np.mean(sa == sb)), 3))              # 0.352 (estimate of 0.375)

```

The `& P61` mask is a cheap stand-in for "mod p"; real implementations use proper modular arithmetic or seeded 64-bit hashes. For banding, use each band's $r$ values as a dictionary key and union documents that share a key.

## Checking that the pipeline helps

Fewer documents is not the goal; better models are.

- **Ablate with small models.** Train identical models of around a billion parameters on tens of billions of tokens with and without the stage, and compare on benchmarks sensitive at that scale (HellaSwag, ARC). FineWeb chose every rule this way.
- **Read samples** of what a rule removed and kept, and **measure removal by slice** (language, domain, code): a rule that removes 60% of code is a hidden mixture decision.
- **Verify dedup**: compute true Jaccard on a sample of candidate pairs.
- **Decontaminate**: search for long n-gram overlaps (13-grams are common) with your evaluation sets.

## What goes wrong

| Problem | Symptom | Fix
| --- | --- | --- |
| Filter thresholds copied from another corpus | a language, code or domain nearly vanishes | measure removal per slice; re-tune by ablation
| Quality classifier too strict | high benchmark scores, narrow and repetitive generations | mix the filtered slice with a larger, less filtered pool
| MinHash threshold too low | many same-topic but distinct pages removed | more rows per band; verify candidates by true Jaccard
| Global dedup across all crawls | old crawls lose most documents; quality of the remainder drops | deduplicate within each snapshot (FineWeb's finding)
| Small sources upweighted without a cap | late-run loss on that source keeps falling while held-out quality stalls; memorisation | cap epochs (around 4) and redistribute the surplus
| Comparing per-token losses across tokenizers | a model looks better only because its tokens are shorter | report bits per byte

## Going deeper

**How accurate is a MinHash estimate?** Each signature position agrees with probability $J$, so with independent hashes the estimate is a binomial proportion with standard deviation $\sqrt{J(1 - J)/k}$. With $k = 112$ that is at most $\sqrt{0.25/112} = 0.047$ (at $J = 0.5$), consistent with the 0.037 spread we measured across pairs spanning many similarity levels.

**Where the threshold comes from.** The S-curve $f(s) = 1 - (1 - s^r)^b$ is steepest roughly where $s^r \approx 1/b$, that is where a single band matches with probability about one in $b$, giving $s^* \approx (1/b)^{1/r}$. For fixed $k = br$, raising $r$ sharpens the step but moves it right; a sharper filter at the same threshold costs more hashes.

**Why per-snapshot dedup can win.** In FineWeb, deduplicating across all 96 snapshots removed most documents from older snapshots, and the remainder it kept was of *lower* quality than what it removed: well-written content gets copied across years, while content unique to one old crawl skews to junk. Every filter changes the *distribution* of what remains, not just its size.

**Data-constrained scaling and selection.** Muennighoff et al. modelled repeated tokens as worth less than fresh ones, with each further epoch worth less again: the quantitative reason for epoch caps. A reference classifier with a threshold is crude importance sampling toward the reference distribution (DSIR makes this explicit with n-gram importance weights), which explains both its gains and its loss of diversity.

## Key terms

| Term | Meaning
| --- | --- |
| Heuristic filter | Hand-written rule on simple document statistics
| Stop words | Common function words; their share separates prose from junk
| Shingle | A run of $n$ consecutive words; a document becomes the set of its shingles
| Jaccard similarity | Shared shingles divided by distinct shingles in either document
| MinHash signature | A document's minimum hash under each of $k$ hash functions
| LSH banding | $b$ bands of $r$ rows; only pairs sharing a band are compared
| Temperature sampling | Mixture weights proportional to $n_i^{1/T}$, flattening the mix as $T$ grows
| Epochs per source | How many times a source is repeated: $p_i D / n_i$
| Bits per byte | Loss normalised by text length in bytes, comparable across tokenizers

<div class="take">
<div class="lab">Key takeaway</div>

treat pretraining data as an engineered pipeline (extract, filter, deduplicate, select, mix, tokenise), validate every rule by small-model ablations rather than by how many documents it removes, and report data budgets together with the tokenizer and the epochs per source.

</div>
