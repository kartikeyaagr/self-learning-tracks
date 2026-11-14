---
id: "06.03"
course: "deep-learning"
module: "06"
slug: "bag-of-words-and-tf-idf"
title: "Bag-of-words and TF-IDF"
keyIdeas: ["A **bag of words** represents a document as a vector of term counts over a fitted vocabulary, ignoring order. Real vectors are over 99% zeros, so they are stored in sparse CSR format.", "**TF-IDF** multiplies how often a term appears in a document by $\\ln\\frac{1+N}{1+\\text{df}} + 1$, so terms frequent here but rare in the corpus score high and terms in every document are shrunk to the minimum weight.", "With L2-normalised TF-IDF rows, cosine similarity is a plain dot product, so ranking a whole collection against a query is one sparse matrix-vector multiply.", "**BM25** improves ranking with term-frequency saturation ($k_1$, the factor never exceeds $k_1 + 1$) and document-length normalisation ($b$). It remains the standard lexical baseline and half of most hybrid search systems.", "Fit the vocabulary and idf on training data only, ideally inside a pipeline. A linear model on TF-IDF unigrams and bigrams is a strong, cheap and interpretable text-classification baseline."]
question: "Under BM25 with k1 = 1.2 and b = 0.75, why doesn't a 10,000-word document that mentions 'python' 20 times automatically outrank a 200-word document that mentions it 5 times, when the average document length is 500 words? Work out the term-frequency factor for each, and say what plain (unnormalised) TF-IDF would have done."
answer: "The BM25 term-frequency factor is $\\text{tf}(k_1+1) / (\\text{tf} + k_1(1 - b + b|d|/\\text{avgdl}))$; the idf part is the same for both documents, so only this factor matters. Long document: $|d|/\\text{avgdl} = 10000/500 = 20$, so the length term is $1.2 \\times (0.25 + 0.75 \\times 20) = 1.2 \\times 15.25 = 18.3$, and the factor is $20 \\times 2.2 / (20 + 18.3) = 44 / 38.3 \\approx 1.15$. Short document: $|d|/\\text{avgdl} = 200/500 = 0.4$, the length term is $1.2 \\times (0.25 + 0.75 \\times 0.4) = 1.2 \\times 0.55 = 0.66$, and the factor is $5 \\times 2.2 / (5 + 0.66) = 11 / 5.66 \\approx 1.94$. The short document scores higher for this term for two reasons. Saturation caps how much repeated mentions can add: whatever tf is, the factor can never exceed $k_1 + 1 = 2.2$, so 20 mentions are not worth four times 5 mentions. Length normalisation recognises that 20 mentions in 10,000 words (0.2%) is a lower density than 5 in 200 (2.5%), so the long document is probably about many things and mentions python in passing. Plain TF-IDF without normalisation would multiply idf by raw tf, giving the long document 20/5 = 4 times the short one's weight; with L2 normalisation the comparison depends on everything else in each document, which is why BM25's explicit, tunable length handling is preferred for ranking."
sources: [{"title": "scikit-learn: Text feature extraction", "url": "https://scikit-learn.org/stable/modules/feature_extraction.html", "note": "CountVectorizer, TfidfVectorizer (exact IDF formula and options) and HashingVectorizer."}, {"title": "Okapi BM25 (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Okapi_BM25", "note": "The BM25 formula, its IDF variants and the roles of k1 and b, with links to the original papers."}, {"title": "Jurafsky and Martin, Speech and Language Processing (3rd ed. draft)", "url": "https://web.stanford.edu/~jurafsky/slp3/", "note": "The vector semantics chapter covers term-document matrices, TF-IDF and cosine similarity."}]
practice: ["tf-idf", "bag-of-words", "document-ranking", "text-classification"]
tags: ["TF-IDF", "BM25", "sparse vectors"]
tools: ["scikit-learn", "NumPy"]
---

## What you will learn

- How a document becomes a **bag of words**: a long vector of word counts with word order thrown away.
- Why those vectors are almost all zeros, and how sparse (CSR) storage makes that cheap.
- How **TF-IDF** turns counts into weights that say what a document is *about*, computed by hand.
- How to rank documents with cosine similarity, and how **BM25** improves the ranking.
- How to use TF-IDF features for classification, and the leaks that quietly spoil results.

*Before you start:* tokenisation (06.01); vectors, dot products and logarithms; logistic regression (Machine Learning course, module 02) for the last step.

The previous lesson (06.02) counted word sequences to predict the next word. This lesson counts words to describe a whole document as one fixed-length vector, so we can compare, search and classify documents. These vectors are what embeddings (06.04) were built to improve on, and still the baseline they must beat.

## The idea in one picture

Imagine a shop's support inbox with four short reviews:

| Doc | Text
| --- | --- |
| 1 | the battery died the first day
| 2 | the screen cracked and the battery swelled
| 3 | the screen is bright and the price is great
| 4 | the delivery was late

The 16 distinct words across all four form the **vocabulary**. Describe each review by 16 numbers, one per vocabulary word, saying how often it appears; stacking the rows gives the **document-term matrix**.

~~~dg-fig 06.03-matrix.svg
Top: raw counts for the four reviews. "the" has the biggest count in every row although it says nothing about the topic. Bottom: after TF-IDF, "the" fades and distinctive words such as died, cracked and late carry the weight.
~~~

In the top panel, doc 1 has a 2 under "the" and a 1 under each of battery, died, first and day. That row is the whole representation: word order is gone. "Dog bites man" and "man bites dog" get the same row, which is why this is called a **bag of words**: the words are tipped into a bag and counted.

The bottom panel shows where this lesson ends up. "the" is in every document, so it cannot tell documents apart; **TF-IDF** shrinks such words and boosts words that are rare across the collection.

## Step 1: build the vocabulary and count

The recipe has two phases, and keeping them apart is the most important habit in this lesson.

1. **Fit**: run the tokeniser over the training documents and assign each distinct term a column number. This mapping from term to column is the vocabulary, with size $V$.
2. **Transform**: for any document (training, test, or a user's search query), count how often each vocabulary term appears and write the counts in a vector of length $V$. Terms that are not in the vocabulary are silently dropped.

In symbols, the count vector of document $d$ is $c_d \in \mathbb{R}^V$ with $c_{d,j} = \text{tf}(t_j, d)$, the **term frequency**: the number of times term $t_j$ occurs in $d$. Our four reviews give a $4 \times 16$ matrix.

A **term** need not be a single word. Adding **n-grams** (runs of $n$ consecutive tokens) as columns lets the model see "not good" or "new york" as one unit, recovering a little order. The vocabulary grows fast, so cap it with `min_df=2` (keep terms seen in at least two documents) or `max_features`.

## Step 2: store it sparsely

Each row has a non-zero only where the document actually uses a word. A 200-word document over a 100,000-term vocabulary therefore has at most 200 non-zeros out of 100,000 entries: at least 99.8% zeros. Even our toy matrix is 66% zeros (22 non-zeros out of 64 cells).

The pattern holds on real text. The figure below uses 1,417 docstrings from Python's standard library (about 84,000 words) as a corpus.

~~~dg-fig 06.03-zipf.svg
Left: word counts fall roughly as 1/rank (Zipf's law), so a handful of words supply most tokens. Right: as documents are added the vocabulary keeps growing (6,186 words here) and the share of zero entries climbs past 99%.
~~~

Word frequencies follow **Zipf's law**: here the 10 most common words supplied 22% of all tokens, while 39% of the vocabulary appeared only once. Each document uses on average 36 distinct words out of 6,186, so 99.4% of the matrix is zero.

A dense float32 matrix for one million documents and a 100,000-term vocabulary needs $10^6 \times 10^5 \times 4$ bytes = 400 GB. The **compressed sparse row (CSR)** format stores only the non-zeros, as three arrays:

- `data`: the non-zero values, row after row;
- `indices`: the column number of each value;
- `indptr`: where each row starts in the other two arrays (length = rows + 1).

With 200 non-zeros per document and 4-byte values plus 4-byte column indices, the same million documents take about 1.6 GB (scikit-learn's default float64 values make it about 2.4 GB).

## Step 3: weight the counts with TF-IDF

Raw counts reward words that are common everywhere. A good weight should be high when a term is frequent *in this document* and rare *in the collection*. TF-IDF is the product of those two ideas:

$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t), \qquad \text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1$$

- $\text{tf}(t, d)$: how many times term $t$ appears in document $d$.
- $N$: the number of documents in the training corpus.
- $\text{df}(t)$: the **document frequency**, the number of documents that contain $t$ at least once (not the total count).
- $\text{idf}(t)$: the **inverse document frequency**, large for rare terms and small for common ones.

This is scikit-learn's default ("smoothed") form. The $+1$ inside acts as if one extra document contained every term, avoiding division by zero. The $+1$ outside keeps a term found in every document at weight 1 instead of 0: shrunk, not deleted. The textbook version is simply $\ln(N/\text{df})$.

~~~dg-fig 06.03-idf.svg
idf falls steadily as a term appears in more documents. A term in 10 of 1,000 documents gets 5.51; one in all 1,000 gets the floor of 1.0 under scikit-learn's formula and 0 under the textbook one.
~~~

Number check with $N = 1{,}000$: "the" is in every document, so $\text{idf} = \ln(1001/1001) + 1 = 1.0$. "transformer" is in 10, so $\text{idf} = \ln(1001/11) + 1 = \ln 91 + 1 = 5.51$. One occurrence of "transformer" outweighs five occurrences of "the" before normalisation.

Two common adjustments:

- **Sublinear tf** replaces $\text{tf}$ by $1 + \ln \text{tf}$ (for $\text{tf} > 0$), so a word used 20 times counts about 4 times as much as a word used once, not 20 times.
- **L2 normalisation** (the default) divides each vector by its length $\|v\| = \sqrt{\sum_j v_j^2}$, so long documents do not dominate and similarity becomes a dot product (Step 4).

<div class="ex">
<div class="lab">Worked example</div>

**TF-IDF by hand for doc 1 and the query**

Corpus: the four reviews, so $N = 4$. Document frequencies: "the" is in 4 documents, "battery" in 2 (docs 1 and 2), "died", "first" and "day" in 1 each.

idf: $\text{idf}(\text{the}) = \ln(5/5) + 1 = 1.000$. $\text{idf}(\text{battery}) = \ln(5/3) + 1 = 0.511 + 1 = 1.511$. $\text{idf}(\text{died}) = \text{idf}(\text{first}) = \text{idf}(\text{day}) = \ln(5/2) + 1 = 0.916 + 1 = 1.916$.

Doc 1 raw weights (tf times idf): the $2 \times 1.000 = 2.000$, battery $1 \times 1.511 = 1.511$, died, first, day $1.916$ each.

Length: $\sqrt{2.000^2 + 1.511^2 + 3 \times 1.916^2} = \sqrt{4.000 + 2.283 + 11.013} = \sqrt{17.296} = 4.159$. Dividing gives the normalised row: the 0.481, battery 0.363, died / first / day 0.461 each, exactly the doc 1 row in the bottom panel of the first figure.

Query "the battery died": raw weights $(1.000, 1.511, 1.916)$, length $\sqrt{1 + 2.283 + 3.671} = \sqrt{6.954} = 2.637$.

Cosine similarity: the dot product of the raw vectors over the product of lengths. Shared terms are the, battery and died: $2.000 \times 1.000 + 1.511 \times 1.511 + 1.916 \times 1.916 = 2.000 + 2.283 + 3.671 = 7.954$. Then $7.954 / (4.159 \times 2.637) = 7.954 / 10.968 = 0.725$.

Notice that "the" still holds weight 0.481 in doc 1 because it appears twice and its idf floor is 1. TF-IDF shrinks stop words relative to rare words; it does not remove them.

</div>

## Step 4: compare documents with cosine similarity

To rank documents for a query, turn the query into a vector with the *same* fitted vocabulary and idf, then measure how closely it points in the same direction as each document. The **cosine similarity** of two vectors $u$ and $v$ is

$$\cos(u, v) = \frac{u \cdot v}{\|u\|\,\|v\|},$$

where $u \cdot v = \sum_j u_j v_j$ is the dot product. It is 1 when the vectors point the same way and 0 when they share no terms (it never goes negative here, since counts are non-negative). Only direction matters, so a long and a short review on one topic can still score high. Once rows are L2-normalised, the denominators are 1 and cosine is just the dot product, so ranking the whole collection is one sparse matrix-vector product: $\text{scores} = X q$.

~~~dg-fig 06.03-ranking.svg
Cosine similarity to "the battery died". With raw counts, the irrelevant delivery review scores 0.29 just for sharing "the"; with TF-IDF it falls to 0.11, and the right answer leads the runner-up by 0.34 instead of 0.24.
~~~

Both put doc 1 first, but TF-IDF separates relevant from irrelevant far more cleanly, and on millions of documents that margin decides whether page one is useful.

~~~dg-flow Lexical search: the same vectoriser, fitted once on the corpus, is applied to documents and queries.
Documents -(tokenise)-> Vocabulary -(count)-> Sparse CSR counts -(× IDF, L2 norm)-> *TF-IDF matrix -(dot with query vector)-> Ranked documents
~~~

## Bag-of-words and TF-IDF in code

scikit-learn's `CountVectorizer` and `TfidfVectorizer` do Steps 1 to 3. Note its default tokeniser keeps only tokens of two or more word characters, so "a" and "I" vanish.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the battery died the first day",
        "the screen cracked and the battery swelled",
        "the screen is bright and the price is great",
        "the delivery was late"]

counts = CountVectorizer().fit_transform(docs)
print(counts.shape, counts.nnz)            # (4, 16) 22  -> 66% zeros even here
print(counts.indptr, counts.indices[:5])   # [ 0  5 11 18 22] [14  1  6  7  4]

vec = TfidfVectorizer()                    # smooth_idf=True, norm="l2", sublinear_tf=False
X = vec.fit_transform(docs)                # learn vocabulary + idf from the corpus
idf = dict(zip(vec.get_feature_names_out(), vec.idf_.round(3)))
print(idf["the"], idf["battery"], idf["died"])     # 1.0 1.511 1.916

q = vec.transform(["the battery died"])    # reuse the SAME vocabulary and idf
scores = (X @ q.T).toarray().ravel()       # rows are unit length -> cosine
print(scores.round(3))                     # [0.725 0.381 0.13  0.109]
print(scores.argsort()[::-1])              # [0 1 2 3]
```

`indptr` says row 0 occupies positions 0 to 4 of `data`/`indices`, row 1 positions 5 to 10: the CSR layout from Step 2. The idf values and scores match the hand calculation.

## Step 5: BM25, TF-IDF tuned for ranking

For search, two weaknesses of TF-IDF show up. Term frequency should **saturate**: the tenth mention of "python" adds less evidence than the second. And long documents mention everything more often, so their counts should be discounted relative to the average length. **BM25** (Okapi "best match" 25) handles both:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{idf}(t) \cdot \frac{\text{tf}(t, d)\,(k_1 + 1)}{\text{tf}(t, d) + k_1\big(1 - b + b\,\frac{|d|}{\text{avgdl}}\big)}$$

- $q$ is the query; the sum runs over its terms.
- $|d|$ is the length of document $d$ in tokens and $\text{avgdl}$ the average length over the corpus.
- $k_1$ (typically 1.2 to 2.0) controls saturation: as tf grows the fraction approaches $k_1 + 1$ and never exceeds it.
- $b$ (typically 0.75) controls length normalisation: $b = 0$ ignores length, $b = 1$ fully scales by relative length.
- The idf is usually $\ln\big(\frac{N - \text{df} + 0.5}{\text{df} + 0.5} + 1\big)$, the variant used by Lucene and Elasticsearch.

~~~dg-fig 06.03-bm25.svg
Left: BM25's term-frequency factor climbs fast and then flattens toward k1 + 1, unlike raw tf, which grows forever. Right: for a term seen 3 times, a document twice the average length gets a smaller factor; larger b means a steeper discount.
~~~

A short NumPy version on the four reviews (doc lengths 6, 7, 9 and 4 tokens):

```python
def bm25(docs, query, k1=1.2, b=0.75):
    cv = CountVectorizer(); tf = cv.fit_transform(docs).toarray()
    N, dl = len(docs), tf.sum(1); avgdl = dl.mean()
    df = (tf > 0).sum(0)
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
    cols = [cv.vocabulary_[t] for t in query.split() if t in cv.vocabulary_]
    f = tf[:, cols]
    norm = k1 * (1 - b + b * dl / avgdl)[:, None]
    return (idf[cols] * f * (k1 + 1) / (f + norm)).sum(1)

print(bm25(docs, "the battery died").round(3))   # [2.107 0.814 0.131 0.125]
```

BM25 scores are not bounded to $[0, 1]$, so compare them only within one query.

BM25 is hard to beat on queries with rare, exact terms: product codes, error messages, names. Dense embedding retrieval (06.04 and the AI Engineering course) wins on paraphrase, where the query and the answer use different words. Production search usually runs both and fuses the ranked lists, for example with reciprocal rank fusion.

## Step 6: TF-IDF as features for classification

A logistic regression or linear SVM on TF-IDF unigrams and bigrams is a strong baseline for topic, spam and sentiment classification. It trains in seconds on a CPU and on long-document topic tasks is often within a few points of much larger fine-tuned models; multinomial naive Bayes on raw counts is cheaper still. Wrap vectoriser and model in one pipeline so vocabulary and idf are learned from the training fold only:

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# 20 hand-written one-line reviews (10 positive, 10 negative), e.g.
# ("great battery and a bright screen", 1), ("screen cracked, not good", 0), ...
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(C=10))
clf.fit(texts, labels)
print(clf.predict(["good screen", "not good screen"]))   # [1 0]
print(clf.predict_proba(["good screen", "not good screen"])[:, 1].round(2))   # [0.72 0.34]
```

~~~dg-fig 06.03-weights.svg
The classifier's largest weights on 20 tiny reviews. The bigram "not good" is the most negative feature, so "not good screen" is classed negative. "all" and "and" sit among the positives purely by accident of this small training set.
~~~

Each weight belongs to one word or phrase, so the model is easy to inspect. The figure also shows the catch: with little data the model latches onto words like "all" (from "not bad at all" and "lasts all week"). On real data the same happens with names, dates or boilerplate, so reading the top weights is a cheap sanity check.

## Checking the result

- **Look at the vocabulary.** Print its size and a sample of terms; thousands of HTML fragments or hex strings mean cleaning needs work.
- **Check density.** `X.nnz / (X.shape[0] * X.shape[1])` should be well under 1%. If the matrix is dense, something like `toarray()` slipped in.
- **Inspect idf extremes.** Lowest idf should be function words and boilerplate; if typos dominate the highest, raise `min_df`.
- **Run known queries.** For search, write 10 to 20 queries with known answers and check they land in the top few.
- **Read the top classifier weights**, as in the last figure, and drop or mask features that are obviously leaking the label.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Fitting the vectoriser on train + test | Validation scores slightly too good; test-set vocabulary and df leak into features | Put the vectoriser inside a `Pipeline` and fit on training data only
| Calling `fit_transform` on the query or test set | Query vectors use a different vocabulary; similarities are nonsense or shapes mismatch | Fit once, then only `transform` new text
| No length normalisation | Long documents rank first for everything | Keep `norm="l2"`, or use BM25 with $b \approx 0.75$
| Huge n-gram range without `min_df` | Vocabulary of millions, memory errors | Cap with `min_df` / `max_features`, or hash
| Converting to dense (`toarray()`) | Memory explodes on a real corpus | Keep CSR; scikit-learn's linear models accept sparse input
| Removing "not", "no" as stop words | Sentiment model cannot tell "good" from "not good" | Check the stop list; keep negations, use bigrams

## Going deeper

**Why the logarithm in idf.** If a term appears in $\text{df}$ of $N$ documents, the probability that a random document contains it is $p = \text{df}/N$, and $\ln(N/\text{df}) = -\ln p$ is the information (in nats) gained by learning the document contains it. Rare terms are surprising, hence informative. Multiplying by tf adds that evidence once per occurrence, the assumption BM25 relaxes with saturation.

**Where BM25 comes from.** BM25 is a practical approximation to the probabilistic relevance framework of Robertson and Spärck Jones: rank by the odds that a document is relevant given its terms. Its saturating tf curve mimics a "2-Poisson" model of how often a term appears in documents that are and are not about it. The BM25 idf goes negative for terms in more than half the documents in the original form; the "+1" inside the logarithm, used by Lucene, keeps it positive.

**The hashing trick.** Hash each term to one of $m$ columns instead of storing a vocabulary (`HashingVectorizer` defaults to $m = 2^{20}$): nothing to fit or ship, and it works on streams, at the cost of collisions and no way back from columns to words. With $V = 100{,}000$ terms in $2^{20} \approx 1.05$ million buckets, the chance a given term shares its bucket with some other term is about $1 - e^{-V/m} = 1 - e^{-0.095} \approx 9\%$. scikit-learn by default gives each term a random sign, so colliding terms partly cancel instead of always adding up.

**Low-rank structure.** A truncated SVD of the TF-IDF matrix (latent semantic analysis, `TruncatedSVD`) compresses documents to a few hundred dense dimensions where co-occurring words like "car" and "automobile" end up close: a bridge to the embeddings of 06.04.

**Complexity.** Building the matrix is linear in the number of tokens. Search engines (Lucene, Elasticsearch) store an **inverted index**, a list of documents per term, so a query touches only documents containing its terms.

## Key terms

| Term | Meaning
| --- | --- |
| Bag of words | A document represented by its term counts, ignoring word order
| Vocabulary | The fitted mapping from each term to a column number
| Term frequency (tf) | How many times a term occurs in one document
| Document frequency (df) | How many documents contain a term at least once
| IDF | A weight that is large for rare terms: $\ln\frac{1+N}{1+\text{df}} + 1$ in scikit-learn
| CSR matrix | Sparse storage of only non-zero values, their columns and row start positions
| Cosine similarity | Dot product of two vectors divided by their lengths; the angle between them
| BM25 | A ranking function adding tf saturation ($k_1$) and length normalisation ($b$) to idf weighting
| Hashing trick | Mapping terms to columns with a hash function instead of a stored vocabulary

<div class="take">
<div class="lab">Key takeaway</div>

bag-of-words and TF-IDF turn text into sparse, interpretable vectors: fit the vocabulary and idf on training data, transform everything else with the same vectoriser, and rank or classify with cheap dot products. BM25 adds term saturation and length normalisation and remains the lexical baseline every dense retriever should be compared against.

</div>
