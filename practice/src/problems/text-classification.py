import numpy as np

META = dict(title='Multinomial naive Bayes text classifier', tags=['NLP', 'Classification'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:04.24'], related=['bag-of-words', 'gaussian-naive-bayes', 'tf-idf'])

STATEMENT = r"""
Multinomial naive Bayes treats a document as a bag of words drawn independently from a class-specific distribution. It trains by counting and is a strong baseline for spam filtering and topic classification.

Write `naive_bayes(train_docs, train_labels, test_docs, alpha=1.0)` that returns `(classes, log_post, preds)`.

## Training
- `classes`: the sorted list of distinct labels. Class $c$ has index $c$ in that list.
- Vocabulary $V$: every distinct token in `train_docs`.
- Prior: $\ln P(c) = \ln(N_c / N)$, where $N_c$ is the number of training documents with label $c$.
- Likelihood with Laplace (add-$\alpha$) smoothing, where $n_{c,w}$ counts occurrences of $w$ in documents of class $c$:
$$P(w \mid c) = \frac{n_{c,w} + \alpha}{\sum_{w'} n_{c,w'} + \alpha |V|}$$

## Prediction
For a test document, ignore tokens that are not in $V$, and compute for every class
$$\text{score}_c = \ln P(c) + \sum_{w \in \text{doc}} \ln P(w \mid c)$$
(repeated tokens count every time). Normalize the scores with a log-softmax over classes to get log-posteriors.

## Output
- `classes`: list of labels.
- `log_post`: float array `(len(test_docs), len(classes))`, each row's exponentials sum to 1.
- `preds`: list of predicted labels, the class with the highest score; ties go to the earlier class in `classes`.
"""

STARTER = '''import numpy as np

def naive_bayes(train_docs, train_labels, test_docs, alpha=1.0):
    """Return (sorted classes, log-posteriors (n_test, n_classes), predicted labels)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def naive_bayes(train_docs, train_labels, test_docs, alpha=1.0):
    classes = sorted(set(train_labels))
    cidx = {c: i for i, c in enumerate(classes)}
    vocab = {w: j for j, w in enumerate(sorted({w for d in train_docs for w in d}))}
    counts = np.zeros((len(classes), len(vocab)))
    ndocs = np.zeros(len(classes))
    for d, y in zip(train_docs, train_labels):
        ndocs[cidx[y]] += 1
        for w in d:
            counts[cidx[y], vocab[w]] += 1
    log_prior = np.log(ndocs / ndocs.sum())
    smoothed = counts + alpha
    log_lik = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))   # row sum = n_c + alpha * |V|

    scores = np.tile(log_prior, (len(test_docs), 1))
    for i, d in enumerate(test_docs):
        ids = [vocab[w] for w in d if w in vocab]                       # unknown words carry no evidence
        scores[i] += log_lik[:, ids].sum(axis=1)
    m = scores.max(axis=1, keepdims=True)
    log_post = scores - (m + np.log(np.exp(scores - m).sum(axis=1, keepdims=True)))
    preds = [classes[k] for k in np.argmax(scores, axis=1)]
    return classes, log_post, preds
'''

HINTS = [
    'Count into a `(classes, vocab)` matrix and a per-class document counter. Smoothing is then `counts + alpha`, normalized along each row.',
    'Work in log space: a document score is the log prior plus a sum of log-likelihood columns, one per known token. Multiplying raw probabilities underflows for long documents.',
    'Log-softmax a row as `s - (max + log(sum(exp(s - max))))`. `np.argmax` picks the first maximum, which matches the tie rule because classes are sorted.',
]

_r = np.random.default_rng(23)
_sport = ["goal", "match", "team", "score", "win", "the", "a"]
_tech = ["code", "gpu", "model", "data", "score", "the", "a"]
_mk = lambda ws, n: [ws[k] for k in _r.integers(0, len(ws), size=n)]
_docs = [_mk(_sport, 8) for _ in range(5)] + [_mk(_tech, 8) for _ in range(6)]
_labels = ["sport"] * 5 + ["tech"] * 6
TESTS = [
    ([["cheap", "pills"], ["meeting", "today"], ["cheap", "meeting"]], ["spam", "ham", "spam"], [["cheap"], ["today"]]),
    ([["a", "a", "b"], ["b", "c"]], [1, 0], [["a", "b", "c", "c"], []], 0.5),
    ([["x"], ["y"]], ["p", "q"], [["unknown", "words"]]),
    ([["x"]], ["only"], [["x"], ["z"]]),
    (_docs, _labels, [_mk(_sport, 10), _mk(_tech, 10), ["score", "the"], _mk(_sport + _tech, 30)]),
    (_docs, _labels, [_mk(_tech, 200)], 0.01),
]

EXPLANATION = r"""
## Why it works
Bayes' rule gives $P(c \mid d) \propto P(c) \prod_w P(w \mid c)$ under the (false but useful) assumption that words are independent given the class. Estimates are just normalized counts, so training is one pass over the data. Laplace smoothing keeps a word that never appeared in a class from forcing that class's probability to exactly zero.

## Log space and normalization
Test 6 scores a 200-word document; the product of 200 probabilities around 0.1 is $10^{-200}$, close to underflow. Sums of logs are exact enough, and log-softmax turns them into posteriors without ever exponentiating a large negative number on its own.

## Complexity
Training is $O(N_{\text{tokens}})$ plus $O(CV)$ for the tables. Prediction costs $O(C)$ per test token.

## Common bugs
- Using the number of documents containing $w$ instead of the number of occurrences (that is the Bernoulli variant).
- Smoothing the denominator with $\alpha$ instead of $\alpha |V|$, so the distribution doesn't sum to 1.
- Giving unknown test words a smoothed probability. Here they are ignored; including them with $\alpha / (\ldots)$ penalizes classes with more training text.

## Interview follow-ups
- Why are naive Bayes posteriors badly calibrated (often extremely close to 0 or 1)? (Correlated words are counted as independent evidence.)
- How does Complement NB help on imbalanced classes?
"""
