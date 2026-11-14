---
id: "08.04"
course: "machine-learning"
module: "08"
slug: "two-tower-retrieval"
title: "Two-tower retrieval"
keyIdeas: ["A two-tower model maps a user (history, context, profile) and an item (ID, text, category) through separate networks to vectors $u, v \\in \\mathbb{R}^d$ and scores $s(u,i) = \\langle u, v \\rangle / \\tau$; because the towers never interact, item vectors can be precomputed and searched with approximate nearest neighbours.", "Training treats retrieval as extreme classification: a softmax over the corpus, approximated by using the **other items in the batch as negatives** (in-batch softmax).", "In-batch negatives are sampled in proportion to popularity, so popular items are over-punished. The **logQ correction** subtracts $\\log q_j$, the item's sampling probability, from each logit to undo that bias.", "Mixing in a few uniformly sampled negatives fixes the opposite gap: rare items never appear as negatives in-batch, so the model never learns to push them down.", "Features in both towers give cold-start coverage that ID-only MF lacks, at the cost of a training pipeline and an ANN index to keep fresh."]
question: "After moving from in-batch-only training to in-batch plus logQ correction, offline recall@200 rises but the share of retrieved items that are new releases (under a week old) falls sharply. Explain the mechanism and propose a fix that keeps the recall gain."
answer: "New releases have very few positives, so their estimated sampling probability $q_j$ is tiny and $-\\log q_j$ is large. Under logQ correction that large term is added to their logits in training, so when a new release is the positive its corrected logit is already high and the loss has little reason to raise its raw score. At serving time, where the correction is not applied, their raw scores end up comparatively low. At the same time, in-batch training rarely shows them as negatives at all, so their region of the space is poorly shaped. Fixes: add uniformly sampled negatives (mixed negative sampling) with their own correction so that tail and new items are contrasted properly; smooth or floor the frequency estimates for young items; and make sure the item tower uses content features, so a new item's vector sits near similar established items instead of depending on a barely trained ID embedding."
sources: [{"title": "Yi et al., Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (2019)", "url": "https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/", "note": "The logQ-corrected in-batch softmax and the streaming frequency estimator."}, {"title": "Yang et al., Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations (2020)", "url": "https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/", "note": "Why in-batch negatives miss the tail, and the fix of adding uniform samples."}, {"title": "Google ML: Candidate retrieval with DNNs", "url": "https://developers.google.com/machine-learning/recommendation/dnn/retrieval", "note": "Short explanation of softmax retrieval, negative sampling and nearest-neighbour serving."}]
practice: ["in-batch-softmax-logq", "uniform-negative-sampling", "clip-cosine-retrieval", "word2vec"]
tags: ["two-tower", "negative sampling"]
tools: ["NumPy"]
---

## Why two towers

Matrix factorisation (08.03) is already a two-tower model with the simplest possible towers: an embedding lookup by ID. The two-tower architecture generalises that by letting each side be a network over features:

- **User/query tower**: recent item IDs (pooled or passed through a small sequence model), user attributes, context such as device and time.
- **Item tower**: item ID embedding plus content features: title text embedding, category, price bucket, age.

The score is a dot product (often of L2-normalised vectors, divided by a temperature $\tau$ around 0.05). The defining constraint is that the two sides meet only at the end. That is what makes retrieval over millions of items feasible: every item vector is computed offline and loaded into an approximate nearest-neighbour index; at request time you compute one user vector and do a top-k search in a few milliseconds.

~~~dg-flow Serving a two-tower retriever: item vectors are built in batch; only the user tower runs per request.
[Offline] Item features -(item tower)-> Item vectors -(build)-> *ANN index
[Online] Request context -(user tower)-> User vector -(top-k search)-> Candidates
~~~

Richer models that cross user and item features early (attention between history and candidate, feature crosses) are more accurate but need a forward pass per candidate, which is why they live in the ranking stage. AI Engineering's "Embeddings and retrieval" module covers ANN indexes (HNSW, IVF, product quantisation) in depth; here the focus is training.

## Training as sampled softmax

Ideally retrieval is trained as a softmax over the whole corpus: given user $u$ who engaged with item $i$,

$$P(i \mid u) = \frac{e^{s(u,i)}}{\sum_{j \in \text{corpus}} e^{s(u,j)}}$$

With 10 million items the denominator is impossible per example. The standard approximation is **in-batch negatives**: in a batch of $B$ (user, positive item) pairs, each user's positive is contrasted with the $B-1$ positives of the other users. One $B \times B$ matrix multiply gives every logit.

```python
import numpy as np
def in_batch_logq_loss(U, V, logq, tau=0.05):
    # U, V: (B, d) L2-normalised user and positive-item vectors
    logits = U @ V.T / tau                 # (B, B), diagonal = positives
    logits = logits - logq[None, :]        # logQ correction per column item
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return -np.mean(np.diag(logp))
```

## Sampling bias and the logQ correction

The in-batch negatives are not uniform: an item appears in a batch in proportion to how often it is a positive, so a hit song that is 1% of all plays is a negative for roughly 1% of all examples, while a niche song almost never is. The gradient therefore pushes popular items down far more than a full softmax would. Without correction, the trained model under-recommends popular items for users who would genuinely like them.

Sampled-softmax theory gives the fix. If negatives are drawn with probability $q_j$, then using corrected logits $s(u,j) - \log q_j$ makes the sampled loss a consistent estimate of the full softmax. Yi et al. (2019, YouTube) applied this to in-batch training and estimated $q_j$ in a streaming way: track, per hashed item ID, the average number of steps between consecutive appearances; $q_j \approx 1/\text{gap}$. The correction is applied only during training. At serving time you rank by the raw $\langle u, v\rangle$.

| Negative source | Bias | Fix
| --- | --- | --- |
| In-batch only | popular items over-penalised, rare items never seen as negatives | logQ correction
| Uniform random from corpus | mostly trivially easy negatives; learns slowly | mix with in-batch; hard negatives
| Mixed (in-batch + uniform) | balances both; extra item-tower passes per batch | correct each source with its own $q$

**Mixed negative sampling** (Yang et al., 2020) appends a few hundred uniformly sampled items to each batch's negatives. Uniform samples cover the long tail, so the model learns that a random obscure item is a bad match, which in-batch training alone never teaches. That matters for item freshness and cold-start items: a brand-new item has a content-feature vector, and you want its position in the space to be meaningful.

## Practical knobs and failure modes

- **Batch size** sets the number of negatives. Going from 512 to 8192 usually helps recall noticeably; distributed training gathers item vectors across devices to enlarge the effective batch.
- **Temperature** $\tau$ controls how sharply the softmax focuses on hard negatives. Too high and everything blurs; too low and training is unstable and over-fits false negatives (items the user would have liked but did not interact with).
- **Accidental hits**: if two users in a batch share the same positive item, one user's positive appears as the other's negative. Mask duplicates of the positive ID.
- **Index freshness**: item vectors depend on the item tower's weights. Retraining the tower without rebuilding the index silently mixes two embedding spaces; version them together.
- **Popularity collapse**: check the share of recommendations coming from the top 1% of items before and after enabling logQ; it should fall towards the share you see in organic engagement.

<div class="ex">
<div class="lab">Worked example</div>

**effect of the correction**

Item A is 2% of positives, item B is 0.001%. In a batch of 4096, A appears as a negative about 82 times per batch in expectation and B about 0.04 times. Their log-probabilities are $\ln 0.02 = -3.9$ and $\ln 0.00001 = -11.5$, so the corrected logits are raw $+3.9$ for A and raw $+11.5$ for B.

When B does turn up as a negative, its inflated logit draws a large gradient; A's many appearances each draw a smaller one. Summed over training, each item is pushed down roughly as much as a full softmax would push it, instead of in proportion to its popularity.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

two-tower retrieval is a sampled softmax over the corpus; in-batch negatives make it cheap, logQ correction makes it unbiased towards popular items, and mixed uniform negatives make it competent on the long tail.

</div>
