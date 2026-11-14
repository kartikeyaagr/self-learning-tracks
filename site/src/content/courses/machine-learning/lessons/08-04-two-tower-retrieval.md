---
id: "08.04"
course: "machine-learning"
module: "08"
slug: "two-tower-retrieval"
title: "Two-tower retrieval"
keyIdeas: ["A two-tower model maps a user (history, context, profile) and an item (ID, text, category) through separate networks to vectors $u, v \\in \\mathbb{R}^d$ and scores $s(u,i) = \\langle u, v \\rangle / \\tau$. Because the towers never interact, item vectors can be precomputed and searched with approximate nearest neighbours.", "Training treats retrieval as a softmax over the whole catalogue, approximated cheaply by using the **other items in the batch as negatives** (in-batch softmax): one $B \\times B$ matrix multiply gives every logit.", "In-batch negatives are drawn in proportion to popularity, so popular items are over-punished. The **logQ correction** subtracts $\\log q_j$, the item's sampling probability, from each training logit to undo that bias; serving uses the raw dot product.", "Mixing in a few uniformly sampled negatives fixes the opposite gap: rare items almost never appear as negatives in-batch, so the model never learns to push them down.", "Temperature decides how much the loss focuses on hard negatives; too low over-fits false negatives. Content features in both towers give cold-start coverage that ID-only MF lacks, at the cost of a training pipeline and an ANN index that must be rebuilt with every tower update."]
question: "After moving from in-batch-only training to in-batch plus logQ correction, offline recall@200 rises but the share of retrieved items that are new releases (under a week old) falls sharply. Explain the mechanism and propose a fix that keeps the recall gain."
answer: "New releases have very few positives, so their estimated sampling probability $q_j$ is tiny and $-\\log q_j$ is large. Under logQ correction that large term is added to their logits in training, so when a new release is the positive its corrected logit is already high and the loss has little reason to raise its raw score. At serving time, where the correction is not applied, their raw scores end up comparatively low. At the same time, in-batch training rarely shows them as negatives at all, so their region of the space is poorly shaped. The recall gain itself is real: the correction stopped over-punishing popular items, which most held-out interactions involve. Fixes that keep it: (1) add uniformly sampled negatives (mixed negative sampling), each source corrected with its own $q$ ($1/N$ for the uniform ones), so that tail and new items are contrasted properly; (2) smooth or floor the frequency estimates for young items, for example by initialising a new item's gap from the average of its category instead of from zero sightings, so a handful of early plays does not produce an extreme correction; (3) make sure the item tower uses content features (title embedding, category, creator), so a new item's vector sits near similar established items instead of depending on a barely trained ID embedding, possibly with ID-embedding dropout during training so the tower learns to work without the ID. Then monitor recall@200 and the new-release share side by side, sliced by item age, rather than recall alone."
sources: [{"title": "Yi et al., Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (2019)", "url": "https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/", "note": "The logQ-corrected in-batch softmax and the streaming frequency estimator."}, {"title": "Yang et al., Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations (2020)", "url": "https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/", "note": "Why in-batch negatives miss the tail, and the fix of adding uniform samples."}, {"title": "Google ML: Candidate retrieval with DNNs", "url": "https://developers.google.com/machine-learning/recommendation/dnn/retrieval", "note": "Short explanation of softmax retrieval, negative sampling and nearest-neighbour serving."}]
practice: ["in-batch-softmax-logq", "uniform-negative-sampling", "clip-cosine-retrieval", "word2vec"]
tags: ["two-tower", "negative sampling"]
tools: ["NumPy"]
---

## What you will learn

- What a two-tower model is, and why keeping the user side and the item side apart until the very end is what makes retrieval over millions of items fast.
- How retrieval is trained as a softmax over the whole catalogue, and how **in-batch negatives** turn that impossible softmax into one matrix multiply.
- Why in-batch negatives are biased towards popular items, and how the **logQ correction** removes that bias, checked by hand on a three-user batch.
- Why a few **uniformly sampled negatives** fix the opposite gap on the long tail, and what temperature and batch size do.
- How to tell whether a trained retriever is healthy, and the failure modes that show up in production.

*Before you start:* you should know matrix factorisation and the dot-product score $x_u^\top y_i$ (08.03), the softmax and cross-entropy loss (02.04), and the retrieve-then-rank funnel from 08.01. Embedding lookups and small feed-forward networks are assumed.

## From factors to towers

Lesson 08.03 learned one vector per user and one per item and scored a pair by their dot product. That is already a two-tower model with the simplest possible towers: a lookup table on each side, meeting only in the final dot product.

The **two-tower model** keeps that shape but lets each side be a real network over features instead of a bare ID lookup. It came from web search (Microsoft's DSSM, 2013) and is now the standard first stage of large recommenders.

## The idea in one picture

Suppose a streaming service has turned one user and eight titles into 2-dimensional vectors of length 1 (real systems use 64 to 256 dimensions). Because all vectors have length 1, the dot product is the cosine of the angle between them: +1 means "pointing the same way", 0 means unrelated, -1 means opposite.

| Title | Space drama | Sci-fi epic | Robot comedy | Heist thriller | Kids cartoon | Romance | Cooking show | Nature doc
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Score with this user | +1.00 | +0.87 | +0.82 | +0.17 | -0.17 | -0.50 | -0.94 | -0.94

~~~dg-fig 08.04-space.svg
The user vector and eight item vectors. Retrieval is "find the items pointing most nearly in the user's direction": the three closest in angle have the highest dot product and come back as candidates.
~~~

Retrieval is now geometry: compute the user's vector, find the items whose vectors point most nearly the same way, and hand the top few hundred to the ranking stage. Three terms:

- The **user tower** (or query tower) is the network that turns everything known about the request into the user vector $u$.
- The **item tower** turns an item's ID and features into the item vector $v$.
- **Approximate nearest-neighbour (ANN) search** finds the top-$k$ dot products among millions of precomputed item vectors in milliseconds, accepting a small chance of missing a true neighbour.

The design hangs on one constraint: **the towers never see each other's inputs**, so every item vector can be computed once, offline, and indexed.

## Step 1: build the two towers

Each tower is usually a few embedding tables feeding a small multi-layer network.

- **User tower inputs:** the IDs of recently watched or bought items (averaged, or passed through a small sequence model), user attributes such as country and language, and request context such as device and time of day.
- **Item tower inputs:** an item ID embedding plus content features: a text embedding of the title or description, category, price bucket, age of the item.

Both towers end in a vector of the same length $d$. The score of user $u$ and item $i$ is

$$s(u, i) = \frac{\langle u, v_i \rangle}{\tau},$$

where $u$ is the user-tower output, $v_i$ the item-tower output for item $i$, $\langle \cdot,\cdot \rangle$ the dot product, and $\tau$ a **temperature**, a small positive number (often around 0.05) that stretches the scores before the softmax. Most systems L2-normalise both vectors first, so $\langle u, v_i\rangle$ is a cosine between -1 and 1 and $\tau$ alone controls how sharp the scores are.

Why bother with content features? Because a brand-new item has no interactions, so a pure ID embedding for it is untrained noise. Its title, category and price, however, exist from minute one, so the item tower can place it near similar established items. That gives two-tower models the **cold-start** coverage that ID-only matrix factorisation lacks.

## Step 2: serve it in two halves

Because the towers do not interact, serving splits cleanly into an offline half and an online half.

~~~dg-flow Serving a two-tower retriever: item vectors are built in batch; only the user tower runs per request.
[Offline] Item features -(item tower)-> Item vectors -(build)-> *ANN index
[Online] Request context -(user tower)-> User vector -(top-k search)-> Candidates
~~~

Per request you run the user tower once and one top-$k$ search. A model that crosses user and item features early (say, attention between history and candidate) is more accurate but needs a forward pass per candidate, so it lives in the ranking stage, which sees only a few hundred items. ANN index types (HNSW graphs, IVF clustering, product quantisation) are covered in AI Engineering's "Embeddings and retrieval" module; the rest of this lesson is about training.

## Step 3: train as a softmax over the catalogue

The training data are logged pairs: user $u$ (with their context at that moment) engaged with item $i$. Ideally we would treat "which item did they pick?" as a classification problem with one class per item:

$$P(i \mid u) = \frac{e^{s(u,i)}}{\sum_{j \in \text{corpus}} e^{s(u,j)}}, \qquad \text{loss} = -\log P(i \mid u).$$

The numerator rewards the true item; the denominator pushes every other item down. The problem is the denominator: with ten million items it needs ten million item-tower passes per training example.

The standard shortcut is **in-batch negatives**. Take a batch of $B$ (user, positive item) pairs. For each user, use the positives of the *other* $B - 1$ users as that user's negatives. One $B \times B$ matrix multiply $U V^\top$ gives every score at once: row $i$ is user $i$ scored against all positives in the batch, and the diagonal holds the true pairs.

~~~dg-fig 08.04-batch.svg
Left: four users scored against the four positive items of their batch. Right: the softmax of each row (τ = 0.5). Training raises the diagonal and lowers everything else, so each user's pair is contrasted with three free negatives.
~~~

This is the same trick CLIP uses for image-text matching. It costs nothing extra, since the item vectors were computed anyway for the positives. But it has a hidden bias.

## Step 4: correct the popularity bias with logQ

Which items end up as negatives? Exactly the items that appear as positives in the batch, so an item is drawn in proportion to how popular it is. A hit title that accounts for 2% of all plays is a negative in about 2% of all batch slots; a niche documentary almost never is. The loss therefore pushes popular items down far more often than a full softmax would.

How bad is it? Take the extreme case: a model that only has a score per item and no user information at all. The full softmax would learn scores that track log popularity. Uncorrected in-batch training learns that every item is equally good, because each item gets pushed down exactly as often as it gets pulled up.

~~~dg-fig 08.04-logq.svg
A bias-only model trained three ways on 30 items with Zipf popularity. Full softmax recovers log popularity; uncorrected in-batch training flattens it completely; adding the logQ correction restores the full-softmax answer.
~~~

Real models have user information, so the damage is partial, but the uncorrected retriever still under-recommends popular items to users who would like them.

The fix comes from **sampled softmax** theory. If negatives are drawn with probability $q_j$, subtract $\log q_j$ from each item's logit during training:

$$s^{c}(u, j) = s(u, j) - \log q_j.$$

Here $q_j$ is the probability that item $j$ appears in a batch slot, which for in-batch sampling is its share of positives. With this correction the sampled loss becomes a consistent estimate of the full softmax. Note two details. The correction is applied to every column, including the positive. And it is used *only in training*: at serving time you rank by the raw $\langle u, v\rangle$.

Why does subtracting a log work? A popular item now appears often as a negative, but each time its logit is lowered by a lot, so it takes only a small share of the push. A rare item appears seldom, but each time its logit is raised a lot, so it takes a large push. Frequency times push per appearance comes out roughly equal to what a full softmax would give.

<div class="ex">
<div class="lab">Worked example</div>

**logQ by hand on a three-user batch**

Batch: Ana's positive is the hit A, Ben's is B, Cai's is C. Similarities: Ana to (A, B, C) = (0.9, 0.3, 0.1); Ben to (A, B, C) = (0.6, 0.8, 0.2). Temperature $\tau = 0.5$, so logits are similarities divided by 0.5.

Sampling probabilities: $q_A = 0.02$, $q_B = q_C = 0.005$. A softmax only cares about differences between logits, so subtracting $\log q_j$ is the same as subtracting $\log(q_j / q_B)$: A loses $\ln 4 = 1.386$, B and C lose nothing.

Ana, uncorrected: logits $(1.8, 0.6, 0.2)$, exponentials $(6.050, 1.822, 1.221)$, sum $9.093$. $P(A) = 6.050 / 9.093 = 0.665$, loss $= -\ln 0.665 = 0.408$.

Ana, corrected: logits $(1.8 - 1.386, 0.6, 0.2) = (0.414, 0.6, 0.2)$, exponentials $(1.512, 1.822, 1.221)$, sum $4.556$. $P(A) = 0.332$, loss $= 1.103$. The hit no longer gets credit for being common, so to win Ana's row the model must raise A's raw score for Ana.

Ben, uncorrected: logits $(1.2, 1.6, 0.4)$, exponentials $(3.320, 4.953, 1.492)$, sum $9.765$. The gradient of the loss on a negative's logit equals its softmax probability, so A, a negative here, is pushed down with weight $3.320 / 9.765 = 0.340$.

Ben, corrected: logits $(-0.186, 1.6, 0.4)$, exponentials $(0.830, 4.953, 1.492)$, sum $7.275$. A's push drops to $0.830 / 7.275 = 0.114$, three times smaller. A is a negative in many rows, so each push must be small; that is the correction at work.

</div>

In a real system nobody hands you $q_j$. Yi et al. (2019, YouTube) estimated it on the fly: keep, per hashed item ID, the training step when the item was last seen and a moving average of the gap between sightings. An item seen every 50 steps has $q_j \approx 1/50$. The estimate adapts as items rise and fall in popularity, with no global counting pass.

## Step 5: add uniform negatives for the long tail

LogQ fixes how hard popular items are pushed, but it cannot push items that never show up. On a catalogue of 100,000 items with Zipf popularity, the 50,000th most popular item appears as an in-batch negative about once every 150 batches, even with a batch of 4,096.

~~~dg-fig 08.04-mixed.svg
How often each item is used as a negative per batch. In-batch sampling follows popularity, so the tail is almost never contrasted. A few hundred uniformly drawn extra negatives give every item a floor.
~~~

So the model never learns that a random obscure item is a poor match for most users, and the tail of the space stays badly shaped. That matters most for new items, which are by definition in the tail.

**Mixed negative sampling** (Yang et al., 2020) appends a few hundred items drawn uniformly from the catalogue to every batch. They are run through the item tower and added as extra columns of the logit matrix, shared by all rows. Each source keeps its own correction: in-batch columns use their popularity-based $q_j$, uniform columns use $q = 1/N$ for a catalogue of $N$ items.

| Negative source | Bias | Fix
| --- | --- | --- |
| In-batch only | popular items over-penalised, rare items never seen as negatives | logQ correction
| Uniform random from corpus | mostly trivially easy negatives; learns slowly | mix with in-batch; hard negatives
| Mixed (in-batch + uniform) | balances both; extra item-tower passes per batch | correct each source with its own $q$

## Step 6: set temperature and batch size

**Batch size** sets how many negatives each example sees. Going from 512 to 8,192 usually lifts recall noticeably; distributed training gathers item vectors across devices to enlarge the batch.

**Temperature** decides which negatives the loss cares about. A negative's gradient weight is proportional to $e^{s/\tau}$, so a small $\tau$ makes the few most similar negatives (the **hard negatives**) take nearly all the push.

~~~dg-fig 08.04-temp.svg
How the push is shared among negatives of different similarity. At τ = 1 even unrelated items get a large share; at τ = 0.05 almost all of it lands on the few negatives closest to the user.
~~~

Too high and training wastes effort on easy items and the space blurs. Too low and training becomes unstable and over-fits **false negatives**: items the user would have liked but simply did not interact with, which the loss treats as wrong answers.

## Two-tower training in code

The NumPy version below is the loss you would write in PyTorch or JAX, plus a brute-force top-$k$. It also masks **accidental hits**: when two users in a batch share the same positive item, that item must not count as a negative for either of them.

```python
import numpy as np

def in_batch_loss(U, V, item_ids, logq=None, tau=0.05):
    # U, V: (B, d) L2-normalised user / positive-item vectors; row i is a true pair
    logits = U @ V.T / tau                          # (B, B), diagonal = positives
    if logq is not None:
        logits = logits - logq[None, :]             # logQ correction, per column item
    same = item_ids[None, :] == item_ids[:, None]   # accidental hits: same item elsewhere
    logits = np.where(same & ~np.eye(len(U), dtype=bool), -np.inf, logits)
    logits -= logits.max(axis=1, keepdims=True)     # numerical safety
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return -np.mean(np.diag(logp))

# the three-user batch from the worked example (cosines given directly, tau = 0.5)
S = np.array([[0.9, 0.3, 0.1], [0.6, 0.8, 0.2], [0.5, 0.0, 0.7]])
U, V = np.eye(3), S.T                                # so that U @ V.T == S
ids = np.array([7, 42, 99])
logq = np.log([0.02, 0.005, 0.005])
print(round(in_batch_loss(U, V, ids, tau=0.5), 3))         # 0.579
print(round(in_batch_loss(U, V, ids, logq, tau=0.5), 3))   # 0.611

# serving: brute-force top-k by raw dot product (an ANN index replaces this at scale)
rng = np.random.default_rng(0)
items = rng.normal(size=(100_000, 32)); items /= np.linalg.norm(items, axis=1, keepdims=True)
u = items[123] + 0.05 * rng.normal(size=32); u /= np.linalg.norm(u)
scores = items @ u
top = np.argpartition(-scores, 10)[:10]
print(top[np.argsort(-scores[top])][:3])                 # [123 971 86701]
```

The losses average all three rows (Cai's included), so they differ from the per-row hand numbers.

## Checking a trained retriever

Training loss depends on batch size and temperature, so it says little. Check these instead.

- **Recall@k on held-out time.** Hold out the last days of interactions and measure the share of true next items that appear in the top 100 or 500 retrieved. Retrieval only has to get the right item *into* the candidate set; ordering is the ranker's job (08.05).
- **Popularity calibration.** Compare the share of retrieved items that come from the top 1% of the catalogue with the share in organic engagement. Without logQ it is usually too low; with logQ it should move towards the organic share.
- **Coverage and freshness.** Track the share of the catalogue ever retrieved, and of items under a week old.
- **Nearest-neighbour sanity checks.** The ten nearest items to a well-known title should make sense to a human.
- **ANN versus exact.** If the ANN top-$k$ misses many of the brute-force top-$k$, tune the index before blaming the model.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Popular items under-retrieved, odd niche results | in-batch training without logQ | subtract $\log q_j$ during training only
| Recall drops after a correct model update | index built with old item-tower weights | version tower and index together; rebuild on every deploy
| New items almost never retrieved | tiny $q_j$, no tail negatives, ID-only item tower | uniform negatives, floor young items' $q$, content features
| Training loss noisy, recall falls as training goes on | temperature too low, false negatives | raise $\tau$, mask accidental hits, de-duplicate pairs
| Good offline recall, flat online metrics | training on clicks that the old system caused | add exploration data, debias logged feedback
| Serving scores differ from offline | logQ applied at serving, or normalisation skipped | raw normalised dot product at serving

The second row is the sneakiest because nothing errors: the system silently compares vectors from two different embedding spaces.

## Going deeper

**Why logQ is the right correction.** Suppose negatives are drawn from $q$ and the model learns scores $s(u,j)$. The sampled loss without correction is minimised when $s(u, j) \approx \log P(j \mid u) - \log q_j + c(u)$: the model learns how much more likely an item is than the sampler would suggest, which is a pointwise mutual information. With $q_j$ proportional to popularity, this cancels exactly the popularity part of $P(j \mid u)$, which is what the bias-only figure showed. Adding $-\log q_j$ inside the training logits shifts the optimum back to $\log P(j \mid u) + c(u)$, the full-softmax answer. The estimate is consistent as the number of negatives grows; for small batches a little bias remains.

**Retrieval as extreme classification.** With one class per item, the item vectors are the class weights, produced by a network rather than stored freely. word2vec's negative sampling applies the same idea to words.

**Dot product versus cosine.** Without normalisation, popular items can win many dot products simply by growing long vectors. Normalising removes that freedom, so popularity must be expressed through direction alone.

**What the architecture cannot do.** The score is a dot product, a rank-$d$ bilinear form. It cannot express "this user likes item A or item B but not both", or any interaction that needs features from both sides at once. That is not a bug to fix in the retriever; it is the reason a ranking stage with feature crosses follows it.

## Key terms

| Term | Meaning
| --- | --- |
| Two-tower model | User and item networks that produce vectors meeting only in a dot product
| User / item tower | The network that maps the request, or one item, to its vector
| ANN search | Fast approximate top-$k$ by dot product over precomputed item vectors
| In-batch negatives | Using the other positives in a batch as each example's negatives
| Sampled softmax | Approximating a full softmax with a sample of negatives
| logQ correction | Subtracting $\log q_j$ from training logits to undo sampling bias
| Mixed negative sampling | Adding uniformly drawn items to in-batch negatives to cover the tail
| Temperature $\tau$ | Divisor on scores; smaller values focus the loss on hard negatives
| Accidental hit | A batch negative that is actually the user's own positive item

<div class="take">
<div class="lab">Key takeaway</div>

two-tower retrieval is a sampled softmax over the catalogue whose two sides meet only in a dot product, so items can be indexed offline; in-batch negatives make training cheap, the logQ correction stops it from punishing popular items, and mixed uniform negatives teach it about the long tail.

</div>
