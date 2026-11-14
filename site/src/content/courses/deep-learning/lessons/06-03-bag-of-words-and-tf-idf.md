---
id: "06.03"
course: "deep-learning"
module: "06"
slug: "bag-of-words-and-tf-idf"
title: "Bag-of-words and TF-IDF"
keyIdeas: ["Bag-of-words represents a document as a vector of term counts over the vocabulary, discarding order. Vectors are huge and about 99.9% zeros, so they are stored sparse.", "TF-IDF reweights counts: terms frequent in this document but rare in the corpus score high; terms in every document score near the minimum.", "With L2-normalised TF-IDF vectors, cosine similarity is a dot product, which makes ranking documents against a query a sparse matrix-vector multiply.", "BM25 improves on TF-IDF for ranking with term-frequency saturation ($k_1$) and document-length normalisation ($b$). It remains the standard lexical baseline and half of most hybrid search systems.", "Fit the vocabulary and IDF on training data only. A linear model on TF-IDF features is a strong, cheap text-classification baseline."]
question: "Under BM25 with k1 = 1.2 and b = 0.75, why doesn't a 10,000-word document that mentions 'python' 20 times automatically outrank a 200-word document that mentions it 5 times, when the average document length is 500 words? Work out the term-frequency factor for each."
answer: "The tf factor is tf(k1+1) / (tf + k1(1 - b + b|d|/avgdl)). Long doc: |d|/avgdl = 20, so the length term is 1.2 × (0.25 + 0.75 × 20) = 1.2 × 15.25 = 18.3; factor = 20 × 2.2 / (20 + 18.3) = 44 / 38.3 ≈ 1.15. Short doc: |d|/avgdl = 0.4, length term = 1.2 × (0.25 + 0.3) = 0.66; factor = 5 × 2.2 / (5 + 0.66) = 11 / 5.66 ≈ 1.94. The short document scores higher for this term: saturation caps how much repeated mentions can add (the factor can never exceed 2.2), and length normalisation recognises that 20 mentions in 10,000 words is a lower density than 5 in 200. Raw TF-IDF without normalisation would have favoured the long document."
sources: [{"title": "scikit-learn: Text feature extraction", "url": "https://scikit-learn.org/stable/modules/feature_extraction.html", "note": "CountVectorizer, TfidfVectorizer (exact IDF formula and options) and HashingVectorizer."}, {"title": "Okapi BM25 (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Okapi_BM25", "note": "The BM25 formula, its IDF variants and the roles of k1 and b, with links to the original papers."}, {"title": "Jurafsky and Martin, Speech and Language Processing (3rd ed. draft)", "url": "https://web.stanford.edu/~jurafsky/slp3/", "note": "The vector semantics chapter covers term-document matrices, TF-IDF and cosine similarity."}]
practice: ["tf-idf", "bag-of-words", "document-ranking", "text-classification"]
tags: ["TF-IDF", "BM25", "sparse vectors"]
tools: ["scikit-learn", "NumPy"]
---

## Documents as count vectors

Before embeddings, the standard way to give text to a model was to count words. Build a vocabulary of $V$ terms from the training corpus; represent each document as a length-$V$ vector whose entry $j$ is how often term $j$ appears. Order is gone ("dog bites man" equals "man bites dog"), which is why this is called a bag of words. The approach is still the first baseline to try for classification and the backbone of lexical search.

Three details matter in practice:

- **Sparsity.** A 200-word document over a 100,000-term vocabulary has at most 200 non-zeros: 99.8% zeros. Store it in a compressed sparse row (CSR) matrix: values, column indices and row pointers. A million documents then take hundreds of megabytes, not the 400 GB a dense float32 matrix would need.
- **N-gram features.** Adding bigrams lets the model see "not good" and "New York" as units. Vocabulary grows fast, so cap it by minimum document frequency.
- **The hashing trick.** Instead of storing a vocabulary, hash each term into one of $2^{20}$ columns. No vocabulary to fit or ship, at the cost of occasional collisions and no way to map columns back to words.

~~~dg-flow Lexical search: the same vectoriser, fitted once on the corpus, is applied to documents and queries.
Documents -(tokenise)-> Vocabulary -(count)-> Sparse CSR counts -(× IDF, L2 norm)-> *TF-IDF matrix -(dot with query vector)-> Ranked documents
~~~

## TF-IDF

Raw counts overweight words that appear everywhere ("the", "of") and say nothing about what a document is about. TF-IDF multiplies a term-frequency part by an inverse-document-frequency part:

$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t), \qquad \text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1$$

where $N$ is the number of documents and $\text{df}(t)$ the number containing $t$. This is scikit-learn's smoothed form: the $+1$ inside avoids division by zero, the $+1$ outside keeps terms in every document from being zeroed out entirely. Common variants:

- **Sublinear tf**: $1 + \ln \text{tf}$ for $\text{tf} > 0$, so a word appearing 20 times does not count 20 times as much as one appearing once.
- **L2 normalisation** of each document vector, so long documents do not dominate and cosine similarity becomes a plain dot product.

<div class="ex">
<div class="lab">Worked example</div>

**IDF for a common and a rare term**

$N = 1{,}000$ documents. "the" appears in all of them: $\text{idf} = \ln(1001/1001) + 1 = 1.0$.

"transformer" appears in 10: $\text{idf} = \ln(1001/11) + 1 = \ln 91 + 1 = 5.51$.

One occurrence of "transformer" outweighs five occurrences of "the" before normalisation.

</div>

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)   # L2 norm by default
X = vec.fit_transform(train_docs)          # fit vocabulary and IDF on training docs only
q = vec.transform(["sparse attention kernels"])
scores = (X @ q.T).toarray().ravel()       # cosine similarity, since rows are unit length
top10 = scores.argsort()[::-1][:10]
```

Fitting on training data only matters: if IDF is computed over train plus test, test-set term statistics leak into the features, mildly inflating evaluation numbers.

## BM25: TF-IDF tuned for ranking

For search, two weaknesses of TF-IDF show up. Term frequency should saturate: the tenth mention of "python" adds less evidence than the second. And long documents mention everything more often, so they should be discounted relative to the average length. BM25 handles both:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{idf}(t) \cdot \frac{\text{tf}(t, d)\,(k_1 + 1)}{\text{tf}(t, d) + k_1\big(1 - b + b\,\frac{|d|}{\text{avgdl}}\big)}$$

- $k_1$ (typically 1.2 to 2.0) controls saturation: as tf grows the fraction approaches $k_1 + 1$, never more.
- $b$ (typically 0.75) controls length normalisation: $b = 0$ ignores length, $b = 1$ fully divides by relative length.
- IDF is usually $\ln\big(\frac{N - \text{df} + 0.5}{\text{df} + 0.5} + 1\big)$.

BM25 is hard to beat on queries with rare, exact terms: product codes, error messages, names. Dense embedding retrieval (lesson 06.04 and the AI Engineering course) wins on paraphrase. Production search usually runs both and fuses the ranked lists, for example with reciprocal rank fusion.

## As features for classification

A linear model (logistic regression or a linear SVM) on TF-IDF unigrams and bigrams is a strong baseline for topic, spam and sentiment classification, often within a few points of fine-tuned transformers on long-document tasks, at a tiny fraction of the cost. Multinomial naive Bayes on raw counts is even cheaper; the Machine Learning course covers both models. Their weights are directly interpretable: the largest coefficients are the words that drive each class.

<div class="take">
<div class="lab">Key takeaway</div>

bag-of-words and TF-IDF turn text into sparse, interpretable vectors that make linear models and lexical search fast. BM25 adds term saturation and length normalisation and remains the lexical baseline every dense retriever should be compared against.

</div>
