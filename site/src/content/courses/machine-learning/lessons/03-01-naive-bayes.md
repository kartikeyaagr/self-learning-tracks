---
id: "03.01"
course: "machine-learning"
module: "03"
slug: "naive-bayes"
title: "Naive Bayes"
keyIdeas: ["A **generative** classifier models $p(y)$ and $p(x \\mid y)$ and classifies with Bayes' rule, $p(y \\mid x) \\propto p(y)\\,p(x \\mid y)$; a discriminative one (logistic regression) models $p(y \\mid x)$ directly.", "Naive Bayes assumes features are **conditionally independent given the class**, so $p(x \\mid y) = \\prod_j p(x_j \\mid y)$: $d$ simple estimates per class instead of an exponential joint table.", "The assumption is usually false, so the probabilities are overconfident (correlated evidence is counted twice), but the argmax is often still right.", "Pick the variant by feature type: Gaussian for continuous, multinomial for counts (text), Bernoulli for binary presence; multinomial NB is a linear classifier in the counts.", "**Smoothing** (add $\\alpha$ to every count) stops one unseen feature from forcing a class probability to exactly zero."]
question: "Through a pipeline bug, every token in your documents is counted twice (each word's count is doubled, and a copied bigram feature duplicates most unigrams). How does this affect a multinomial naive Bayes classifier's predicted classes and probabilities, compared with an L2-regularised logistic regression trained on the same features?"
answer: "For naive Bayes, the log-posterior is $\\log\\pi_k + \\sum_j x_j\\log\\theta_{kj}$. Doubling every count doubles the likelihood term while the prior term stays the same, so the evidence now outweighs the prior twice as heavily. Predictions can flip toward the class the words favour, especially for rare classes whose prior was holding them back. Posteriors are pushed further toward 0 and 1, because each likelihood ratio is effectively squared. Duplicated features have the same effect, since their evidence is counted twice. Logistic regression learns its weights by maximising conditional likelihood. If the counts double, it can halve its weights; if a feature is duplicated, L2 splits the weight between the copies. In both cases its predictions and probabilities stay roughly unchanged. Discriminative training corrects for redundancy; naive Bayes, by assumption, cannot."
sources: [{"title": "On Discriminative vs. Generative Classifiers", "url": "https://papers.nips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html", "note": "Ng and Jordan (NeurIPS 2001): naive Bayes converges faster, logistic regression ends lower."}, {"title": "Cornell CS4780 lecture notes: Bayes Classifier and Naive Bayes", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html", "note": "Derivation of the variants and the proof that multinomial NB is linear."}, {"title": "scikit-learn: Naive Bayes", "url": "https://scikit-learn.org/stable/modules/naive_bayes.html", "note": "Gaussian, multinomial, complement and Bernoulli NB, with their smoothing parameters."}]
practice: ["text-classification", "gaussian-naive-bayes", "bayes-theorem", "n-gram-language-model", "bag-of-words"]
tags: ["generative models", "smoothing"]
tools: ["NumPy", "scikit-learn"]
---

## Where this module goes

Module 02 fitted $p(y \mid x)$ or a decision function directly. Those are **discriminative** models. This module covers two different families. **Generative** classifiers model how each class produces its inputs, $p(x \mid y)$, together with class frequencies $p(y)$, and turn that into a prediction with Bayes' rule: naive Bayes here, discriminant analysis in 03.03. **Instance-based** methods skip the model and compare a query to stored examples: k-nearest neighbours in 03.02. The module ends with calibration (03.04), which asks whether any classifier's probabilities mean what they say. Bayes' rule itself is covered in ML Maths (lesson 10.01).

## Generative classification and the independence assumption

Bayes' rule gives

$$p(y = k \mid x) = \frac{p(y = k)\,p(x \mid y = k)}{\sum_c p(y = c)\,p(x \mid y = c)}.$$

The prior $p(y = k)$ is just the class frequency. The hard part is the class-conditional density $p(x \mid y)$. With $d = 30$ binary features, a full table needs $2^{30} - 1 \approx 10^9$ probabilities per class, far more than any dataset can estimate.

Naive Bayes makes one strong assumption: given the class, features are independent,

$$p(x \mid y = k) = \prod_{j=1}^{d} p(x_j \mid y = k).$$

Now each class needs $d$ one-dimensional estimates, which take a single pass over the data to compute: counting, no optimisation. Classify by the largest log-posterior, $\log p(y = k) + \sum_j \log p(x_j \mid y = k)$; working in logs avoids multiplying hundreds of small numbers into underflow.

~~~dg-flow Naive Bayes fits by counting and predicts by adding log-probabilities, one term per feature.
Training data -(count per class)-> Priors and per-feature likelihoods -(new x: sum logs)-> Log-posterior per class -(argmax)-> *Prediction
~~~

## Why it works anyway, and where it doesn't

Words in an email are not independent given "spam": "free" and "offer" co-occur. So why does naive Bayes classify text well? Classification only needs the right class to have the highest score, not correct probabilities. Correlated features push the score in the same direction more than once, which inflates its size but often leaves its sign intact.

The cost shows up in the probabilities. If one feature is duplicated, its log-likelihood ratio is added twice, which is the same as squaring its likelihood ratio. Posteriors get pushed toward 0 and 1, so naive Bayes probabilities are typically overconfident and need calibration (03.04) before you use them as probabilities.

Ng and Jordan compared naive Bayes with logistic regression, its discriminative counterpart. Naive Bayes approaches its best error with far fewer examples (on the order of $\log d$ rather than $d$), but logistic regression usually reaches a lower error once data is plentiful. That is the generative trade in general: a stronger assumption buys sample efficiency at the price of bias when the assumption is wrong. With a few hundred labelled documents, naive Bayes is a strong baseline; with a million, logistic regression on the same features usually wins.

## Variants by feature type

| Variant | Feature type | Model of p(x_j given y) | Typical use
| --- | --- | --- | --- |
| Gaussian | continuous | normal with per-class mean and variance | sensor or tabular features
| Multinomial | counts | word-frequency distribution over the vocabulary | text classification with counts or TF-IDF
| Bernoulli | binary | presence or absence per feature | short texts, binary flags
| Complement | counts | fitted on all other classes | imbalanced text classes

Two facts help in interviews and in debugging. **Multinomial NB is linear**: its log-posterior is $\log\pi_k + \sum_j x_j\log\theta_{kj}$, a weight per word times its count. It is a linear classifier with weights set by counting instead of by optimisation. **Bernoulli NB also penalises absence**: a word typical of spam that is missing from a message lowers the spam score, which multinomial NB ignores. And Gaussian NB is QDA with diagonal covariance matrices, which ties it to lesson 03.03.

## Smoothing

The maximum-likelihood estimate for multinomial NB is $\theta_{kj} = N_{kj}/N_k$, the share of word $j$ among all words in class $k$. If a word never appeared in spam during training, $\theta = 0$, its log is $-\infty$, and any message containing it can never be spam, however much other evidence there is. Additive smoothing fixes this:

$$\theta_{kj} = \frac{N_{kj} + \alpha}{N_k + \alpha d}.$$

$\alpha = 1$ is Laplace smoothing, $\alpha < 1$ is Lidstone. It equals the posterior mean under a symmetric Dirichlet prior. Tune $\alpha$ by cross-validation on a log grid; for large vocabularies, values like 0.01 to 0.1 often beat 1.

```python
import numpy as np
def fit_mnb(X, y, alpha=1.0):          # X: (n, d) word counts, y: (n,) labels
    classes = np.unique(y)
    log_prior = np.log([np.mean(y == c) for c in classes])
    counts = np.stack([X[y == c].sum(axis=0) for c in classes]) + alpha
    log_theta = np.log(counts / counts.sum(axis=1, keepdims=True))
    return classes, log_prior, log_theta

def predict_mnb(model, X):
    classes, log_prior, log_theta = model
    return classes[np.argmax(X @ log_theta.T + log_prior, axis=1)]   # linear in counts
```

<div class="ex">
<div class="lab">Worked example</div>

**a four-word spam filter**

Vocabulary: free, win, meeting, report. Training word counts are spam (3, 2, 0, 1), total 6, and ham (1, 0, 4, 3), total 8. Priors: spam 0.4, ham 0.6.

With $\alpha = 1$ and $d = 4$: spam $\theta = (4, 3, 1, 2)/10 = (0.4, 0.3, 0.1, 0.2)$; ham $\theta = (2, 1, 5, 4)/12 = (0.167, 0.083, 0.417, 0.333)$.

Message "free win meeting": spam score $0.4 \cdot 0.4 \cdot 0.3 \cdot 0.1 = 0.0048$; ham score $0.6 \cdot 0.167 \cdot 0.083 \cdot 0.417 = 0.0035$. Normalised, $p(\text{spam}) = 0.58$.

Without smoothing, spam's count for "meeting" is 0, so the spam score is exactly 0 and the filter is certain the message is ham, despite "free" and "win".

</div>

<div class="take">
<div class="lab">Key takeaway</div>

naive Bayes is a generative classifier that trains by counting under a conditional-independence assumption. It is fast, data-efficient and a strong text baseline, but smooth the counts and don't trust its probabilities without calibration.

</div>
