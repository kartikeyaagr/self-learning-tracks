---
id: "06.04"
course: "deep-learning"
module: "06"
slug: "word-embeddings"
title: "Word embeddings"
keyIdeas: ["A word embedding is a dense vector (typically 100-300 dimensions) learned so that words appearing in similar contexts get similar vectors: the distributional hypothesis made trainable.", "Skip-gram with negative sampling trains a logistic classifier to tell real (word, context) pairs from random ones. It replaces a $V$-way softmax with $k + 1$ binary decisions per pair.", "GloVe fits word vectors so their dot products match log co-occurrence counts, by weighted least squares on the global count matrix.", "fastText represents a word as the sum of its character n-gram vectors, so it can build vectors for misspellings and unseen inflections.", "Static embeddings give one vector per word type, so 'bank' (river) and 'bank' (money) share one vector. Contextual models (lesson 06.06) fix that."]
question: "You need word vectors for a domain of clinical notes full of rare drug names, abbreviations and misspellings. Compare word2vec and fastText for this, including one way fastText could mislead you."
answer: "word2vec learns one vector per vocabulary entry, so misspellings and rare drug names either fall below the minimum count and get no vector, or get noisy vectors from very few contexts; every unseen variant at test time has nothing. fastText builds each word from character n-gram vectors, so 'amoxicilin' (misspelt) shares most n-grams with 'amoxicillin' and lands near it, and inflections and unseen words get usable vectors. It also shares statistical strength across words with common pieces. The risk is that surface similarity is not semantic similarity: drug names that differ by a few characters but are clinically unrelated (sound-alike or look-alike medications) will get similar vectors, and short abbreviations with few n-grams can collide. Check nearest neighbours for critical terms before trusting them."
sources: [{"title": "Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)", "url": "https://arxiv.org/abs/1301.3781", "note": "Introduces skip-gram and CBOW."}, {"title": "Distributed Representations of Words and Phrases (Mikolov et al., 2013)", "url": "https://arxiv.org/abs/1310.4546", "note": "Negative sampling, the 3/4-power noise distribution and frequent-word subsampling."}, {"title": "GloVe: Global Vectors for Word Representation (Pennington et al., 2014)", "url": "https://aclanthology.org/D14-1162/", "note": "The weighted least-squares objective and its motivation from co-occurrence ratios."}, {"title": "Enriching Word Vectors with Subword Information (Bojanowski et al., 2017)", "url": "https://arxiv.org/abs/1607.04606", "note": "fastText's character n-gram model and results on rare words."}]
practice: ["word2vec", "glove", "fasttext", "embedding-similarity", "cosine-similarity-matrix"]
tags: ["word embeddings", "word2vec"]
tools: ["NumPy"]
---

## From sparse counts to dense vectors

The previous lesson's vectors are sparse and know nothing about meaning: "car" and "automobile" are orthogonal. Word embeddings learn a dense vector per word so that similarity in the space reflects similarity in usage. The idea behind every method here is the distributional hypothesis: words that occur in similar contexts tend to have similar meanings. The AI Engineering course covers embeddings as a retrieval tool; this lesson covers how word vectors are trained.

Mechanically, an embedding is a $V \times D$ matrix and looking up a word is selecting a row. That is the same as multiplying a one-hot vector by the matrix, but far cheaper. The gradient of a lookup is sparse: only the rows used in the batch receive updates, and a word that appears twice in a batch gets both contributions added.

## word2vec: skip-gram with negative sampling

Skip-gram (Mikolov et al., 2013) slides a window over the corpus. For each centre word $c$ and each context word $o$ within, say, 5 positions, it wants the model to predict $o$ from $c$. Each word has two vectors: $v_w$ when it is the centre and $u_w$ when it is the context. The full softmax $P(o \mid c) = \exp(u_o^\top v_c) / \sum_w \exp(u_w^\top v_c)$ needs a sum over the whole vocabulary for every pair, too slow at $V = 10^6$.

Negative sampling replaces it with a binary task: is this pair real or noise? For each real pair, sample $k$ noise words $n_1..n_k$ and maximise

$$\log \sigma(u_o^\top v_c) + \sum_{j=1}^{k} \log \sigma(-u_{n_j}^\top v_c)$$

~~~dg-flow One skip-gram negative-sampling update for the centre word "fox".
Centre "fox" -(lookup)-> v(fox) -(dot with u(quick), u(k noise words))-> *k+1 sigmoid scores -(1 real, k fake)-> Logistic loss
~~~

The details that make it work:

- $k$ = 5 to 20 for small corpora, 2 to 5 for large ones.
- Noise words are drawn from the unigram distribution raised to the 3/4 power, which samples rare words more often than their raw frequency.
- Frequent words are subsampled: each occurrence of $w$ is dropped with probability $1 - \sqrt{t / f(w)}$, $t \approx 10^{-5}$, so "the" stops dominating the updates and the effective window widens.
- CBOW is the mirror variant: predict the centre word from the average of its context vectors. Faster; skip-gram is better for rare words.

<div class="ex">
<div class="lab">Worked example</div>

**what the 3/4 power does**

"the" has frequency 0.05; a rare word has frequency $10^{-6}$. Raw ratio: 50,000 to 1.

After the 3/4 power: $50{,}000^{0.75} \approx 3{,}340$ to 1. Rare words become about 15 times more likely to appear as negatives, so their vectors get pushed away from unrelated words more often.

</div>

```python
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
def sgns_step(V_in, U_out, c, o, negs, lr=0.025):
    v = V_in[c]                                   # (D,)
    ids = np.concatenate([[o], negs])             # 1 positive, k negatives
    labels = np.r_[1.0, np.zeros(len(negs))]
    g = sig(U_out[ids] @ v) - labels              # d loss / d score, (k+1,)
    grad_v = g @ U_out[ids]
    np.subtract.at(U_out, ids, lr * g[:, None] * v)   # handles repeated ids
    V_in[c] -= lr * grad_v
```

## GloVe

GloVe (Pennington et al., 2014) starts from the global co-occurrence matrix $X$, where $X_{ij}$ counts how often word $j$ appears in the context of word $i$, and fits

$$J = \sum_{i,j:\,X_{ij} > 0} f(X_{ij})\,\big(w_i^\top \tilde w_j + b_i + \tilde b_j - \log X_{ij}\big)^2, \quad f(x) = \min\big(1, (x/x_{\max})^{0.75}\big)$$

with $x_{\max} = 100$. The weighting stops very frequent pairs from dominating and ignores zero counts. The final vector is usually $w + \tilde w$. In practice skip-gram and GloVe give similar quality; GloVe trains on a precomputed matrix, skip-gram streams over text.

## fastText

Both methods above treat each word as an atom, so "running", "runner" and the typo "runnning" share nothing, and an unseen word has no vector. fastText (Bojanowski et al., 2017) represents a word as the sum of vectors for its character n-grams (lengths 3 to 6, with boundary markers, e.g. `<ru`, `run`, `unn`, ... plus the whole word), hashed into about 2 million buckets. Training is otherwise skip-gram. Any string, including an out-of-vocabulary word, gets a vector from its n-grams, which matters for morphologically rich languages and noisy text.

| Method | Trains on | Unit | Unseen words |
| --- | --- | --- | --- |
| word2vec (SGNS) | streaming windows | word | no vector |
| GloVe | global co-occurrence counts | word | no vector |
| fastText | streaming windows | character n-grams | built from n-grams |

## Similarity, analogies and limits

Compare vectors with cosine similarity: normalise to unit length, then dot. The famous analogy, $v_{king} - v_{man} + v_{woman} \approx v_{queen}$, holds only if you exclude the three input words from the nearest-neighbour search; otherwise "king" itself often wins. Treat analogy benchmarks as a sanity check, not a quality measure.

Known limits:

- **One vector per type**: "bank" gets a blend of its senses. Contextual models (06.06) produce a different vector per occurrence.
- **Frequency effects**: very frequent and very rare words cluster in odd regions; rare words have noisy vectors.
- **Social bias**: vectors absorb associations from the corpus (gender-occupation analogies are the classic demonstration) and pass them into downstream models.

<div class="take">
<div class="lab">Key takeaway</div>

word embeddings compress co-occurrence statistics into dense vectors where cosine similarity tracks usage. word2vec and GloVe learn one vector per word; fastText builds vectors from subword pieces; all are static, which is the limitation contextual models remove.

</div>
