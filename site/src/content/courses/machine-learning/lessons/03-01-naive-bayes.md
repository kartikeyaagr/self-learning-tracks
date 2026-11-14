---
id: "03.01"
course: "machine-learning"
module: "03"
slug: "naive-bayes"
title: "Naive Bayes"
keyIdeas: ["A **generative** classifier models $p(y)$ and $p(x \\mid y)$ and classifies with Bayes' rule, $p(y \\mid x) \\propto p(y)\\,p(x \\mid y)$; a discriminative one (logistic regression) models $p(y \\mid x)$ directly.", "Naive Bayes assumes features are **conditionally independent given the class**, so $p(x \\mid y) = \\prod_j p(x_j \\mid y)$: $d$ simple estimates per class instead of an exponential joint table. Training is just counting.", "Predict by adding logs, $\\log p(y) + \\sum_j \\log p(x_j \\mid y)$, to avoid underflow; **smoothing** (add $\\alpha$ to every count) stops one unseen feature from forcing a class probability to exactly zero.", "Pick the variant by feature type: Gaussian for continuous, multinomial for counts (text), Bernoulli for binary presence (it also penalises absence); multinomial NB is a linear classifier in the counts.", "The assumption is usually false, so correlated evidence is counted more than once: the argmax is often still right, but probabilities are overconfident and need calibration.", "Naive Bayes learns fast from few examples; with plenty of data logistic regression on the same features usually reaches lower error."]
question: "Through a pipeline bug, every token in your documents is counted twice (each word's count is doubled, and a copied bigram feature duplicates most unigrams). How does this affect a multinomial naive Bayes classifier's predicted classes and probabilities, compared with an L2-regularised logistic regression trained on the same features?"
answer: "For naive Bayes, the log-posterior of class $k$ is $\\log\\pi_k + \\sum_j x_j\\log\\theta_{kj}$. Doubling every count doubles the likelihood term while the prior term stays the same, so the evidence now outweighs the prior twice as heavily. (The estimated $\\theta_{kj}$ barely change, since doubling all training counts leaves their ratios the same apart from the relative effect of smoothing, which halves.) Predictions can therefore flip toward the class the words favour, especially for rare classes whose prior was holding them back, and documents near the boundary move with them. Posteriors are pushed further toward 0 and 1, because each likelihood ratio is effectively squared: with equal priors, a message the old model rated 80% spam (log-odds 1.39) is now rated about 94% (log-odds 2.77). Duplicated bigram features have the same effect, since their evidence is counted twice; naive Bayes has no mechanism to notice that two features say the same thing, because each weight is estimated from its own counts alone. Logistic regression learns its weights jointly by maximising conditional likelihood. If the counts double, it can simply halve its weights; if a feature is duplicated, L2 splits the weight between the copies, giving the same total effect. In both cases its predictions and probabilities stay roughly unchanged. Practical response: fix the pipeline, and in general check naive Bayes probabilities with a reliability diagram or log loss on held-out data and calibrate them (03.04) before using them as probabilities. Discriminative training corrects for redundancy; naive Bayes, by assumption, cannot."
sources: [{"title": "On Discriminative vs. Generative Classifiers", "url": "https://papers.nips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html", "note": "Ng and Jordan (NeurIPS 2001): naive Bayes converges faster, logistic regression ends lower."}, {"title": "Cornell CS4780 lecture notes: Bayes Classifier and Naive Bayes", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html", "note": "Derivation of the variants and the proof that multinomial NB is linear."}, {"title": "scikit-learn: Naive Bayes", "url": "https://scikit-learn.org/stable/modules/naive_bayes.html", "note": "Gaussian, multinomial, complement and Bernoulli NB, with their smoothing parameters."}]
practice: ["text-classification", "gaussian-naive-bayes", "bayes-theorem", "n-gram-language-model", "bag-of-words"]
tags: ["generative models", "smoothing"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- What a **generative** classifier is, and how Bayes' rule turns "what does each class look like?" into "which class is this?".
- The one "naive" assumption that makes the method possible, and why it shrinks a billion-entry table to a few dozen numbers.
- How to train naive Bayes by counting, why the counts need **smoothing**, and why predictions are made by adding logarithms.
- Which variant to use for which kind of feature: Gaussian, multinomial, Bernoulli or complement.
- Why naive Bayes often picks the right class but reports badly overconfident probabilities, and when logistic regression beats it.

*Before you start:* you need conditional probability and Bayes' rule (ML Maths 10.01), logarithms, and the idea of a classifier that outputs class probabilities. Logistic regression (02.03) is the comparison point throughout, so have it fresh in mind.

## Where this module starts

Module 02 fitted $p(y \mid x)$, or a decision function, directly. Those are **discriminative** models: they learn the boundary between classes and nothing else. This module covers two different families.

**Generative** classifiers model how each class produces its inputs, $p(x \mid y)$, and how common each class is, $p(y)$, then predict with Bayes' rule: naive Bayes here, discriminant analysis in 03.03. **Instance-based** methods skip the model altogether and compare a query to stored examples: k-nearest neighbours in 03.02. The module ends with calibration (03.04), which asks whether any classifier's probabilities mean what they say.

## The idea in one picture

You want to filter spam. You have ten training emails, four spam and six ham (ham = legitimate mail), and to keep the numbers tiny you only track four words. Adding up the word counts over all emails of each class gives:

| Class | free | win | meeting | report | total words | emails
| --- | --- | --- | --- | --- | --- | --- |
| Spam | 3 | 2 | 0 | 1 | 6 | 4
| Ham | 1 | 0 | 4 | 3 | 8 | 6

Naive Bayes turns each row into a "profile": how likely each vocabulary word is in that class. A new message goes to the class whose profile makes its words more likely, with a head start for the more common class.

~~~dg-fig 03.01-words.svg
Left: each class's word profile (after smoothing, explained in step 3). "free" and "win" are typical of spam, "meeting" and "report" of ham. Right: scoring "free win meeting" starts from the prior (ham is more common, so below zero), then each word pushes the score up or down by its own fixed amount.
~~~

Read the right panel as a tug of war. The score starts at $-0.41$ because ham is more common. "free" adds $+0.88$, "win" adds $+1.28$, "meeting" subtracts $1.43$. The final score $+0.32$ is above zero, so spam wins narrowly, with probability about 58%.

- The **prior** $p(y)$ is how common a class is before you look at the input (spam 0.4, ham 0.6).
- The **likelihood** $p(x \mid y)$ is how probable this input is if it came from that class.
- The **posterior** $p(y \mid x)$ is what you actually want: the probability of the class *given* the input.

## Step 1: Bayes' rule flips the question

A discriminative model learns $p(y \mid x)$ directly. A generative model learns the other two pieces and combines them with Bayes' rule:

$$p(y = k \mid x) = \frac{p(y = k)\,p(x \mid y = k)}{\sum_c p(y = c)\,p(x \mid y = c)}.$$

Here $k$ is one class and the denominator (the **evidence**) sums over all classes $c$. It is the same for every $k$ and only rescales scores to add to 1, so to choose a class you compare numerators:

$$\text{score}_k = p(y = k)\,p(x \mid y = k).$$

Number check: if a message has likelihood 0.012 under spam and 0.005 under ham, the scores are $0.4 \times 0.012 = 0.0048$ and $0.6 \times 0.005 = 0.003$. Spam wins, and its posterior is $0.0048 / (0.0048 + 0.003) = 0.615$.

The prior is just the class fraction. The hard part is the likelihood, a probability over whole inputs.

## Step 2: the naive assumption

Why is $p(x \mid y)$ hard? Suppose each email is described by $d = 30$ yes/no features ("contains free", "contains win", ...). There are $2^{30} \approx 10^9$ possible emails, so a full table of $p(x \mid y)$ needs about a billion probabilities per class. No dataset is big enough to estimate that.

Naive Bayes makes one strong assumption: **once you know the class, the features carry no information about each other**. This is called **conditional independence given the class**, and it lets the joint probability split into a product of one-feature pieces:

$$p(x \mid y = k) = \prod_{j=1}^{d} p(x_j \mid y = k).$$

Here $x_j$ is the $j$-th feature and $\prod$ means "multiply over $j = 1, \dots, d$". Each factor is a one-dimensional distribution that is easy to estimate. The 30-feature example now needs 30 numbers per class instead of a billion.

The assumption is almost always false: within spam, "free" and "offer" turn up together far more often than independence predicts. Hence *naive*. Later sections show why it works anyway and what it costs.

## Step 3: estimate each piece by counting

Training involves no optimisation: every quantity is a count divided by a count. For word counts (the **multinomial** variant), the estimate for class $k$ and word $j$ is

$$\theta_{kj} = \frac{N_{kj}}{N_k},$$

where $N_{kj}$ is how many times word $j$ appears in all training text of class $k$, and $N_k$ is the total number of words in class $k$. In the table, spam has $\theta_{\text{spam},\,\text{free}} = 3/6 = 0.5$.

This raw estimate has a fatal flaw. "meeting" never appeared in spam, so its estimate is $0/6 = 0$. Because the likelihood is a *product*, any message containing "meeting" could then never be spam, however many spammy words it has.

The fix is **additive smoothing**: pretend every word was seen $\alpha$ extra times in every class,

$$\theta_{kj} = \frac{N_{kj} + \alpha}{N_k + \alpha d},$$

where $d$ is the vocabulary size (the denominator grows by $\alpha d$ so the probabilities still add to 1). $\alpha = 1$ is **Laplace smoothing**; $\alpha < 1$ is called Lidstone smoothing. Number check with $\alpha = 1$, $d = 4$: spam's "meeting" becomes $(0 + 1)/(6 + 4) = 0.1$. Small, but no longer zero.

## Step 4: predict by adding logarithms

A long email might contribute 500 factors of about 0.001 each. Their product, around $10^{-1500}$, is far below the smallest positive 64-bit float (about $10^{-308}$), so it rounds to exactly 0 for every class. This is **underflow**.

The fix is to take logs. The log turns a product into a sum and keeps the ranking unchanged (bigger score, bigger log). The prediction rule becomes

$$\hat y = \arg\max_k \Big[\log p(y = k) + \sum_{j=1}^{d} \log p(x_j \mid y = k)\Big].$$

$\arg\max_k$ means "the class $k$ with the largest value". Each feature adds its own term, just like the steps in the right panel of the first figure.

~~~dg-flow Naive Bayes fits by counting and predicts by adding log-probabilities, one term per feature.
Training data -(count per class)-> Priors and per-feature likelihoods -(new x: sum logs)-> Log-posterior per class -(argmax)-> *Prediction
~~~

For probabilities, subtract the largest log score, exponentiate and normalise (the **log-sum-exp** trick); `predict_proba` does this for you.

<div class="ex">
<div class="lab">Worked example</div>

**the four-word spam filter by hand**

Data from the table: spam counts $(3, 2, 0, 1)$, total 6; ham counts $(1, 0, 4, 3)$, total 8. Priors: spam $4/10 = 0.4$, ham $6/10 = 0.6$. Smoothing $\alpha = 1$, vocabulary $d = 4$, so each denominator grows by 4.

Spam: $\theta = (3{+}1, 2{+}1, 0{+}1, 1{+}1)/10 = (0.4, 0.3, 0.1, 0.2)$. Ham: $\theta = (2, 1, 5, 4)/12 = (0.167, 0.083, 0.417, 0.333)$. Each row sums to 1.

Message "free win meeting". Spam score: $0.4 \times 0.4 \times 0.3 \times 0.1 = 0.0048$. Ham score: $0.6 \times 0.167 \times 0.083 \times 0.417 = 0.0035$ (exactly $0.6 \times 10/1728 = 0.00347$).

Normalise: $p(\text{spam} \mid x) = 0.0048 / (0.0048 + 0.00347) = 0.58$. The filter says spam, but only just.

Same thing in logs: $\log 0.0048 = -5.34$ and $\log 0.00347 = -5.66$. The difference $0.32$ is the final point in the figure, and $1/(1 + e^{-0.32}) = 0.58$ recovers the same probability.

Without smoothing, spam's "meeting" probability is $0/6 = 0$ and ham's "win" probability is $0/8 = 0$. Both scores are exactly zero, so the filter cannot compare them at all (in logs, $-\infty$ against $-\infty$). For "free win report" the unsmoothed ham score is 0 while spam's is positive, so the filter would be 100% certain of spam on the strength of one word never seen in ham.

</div>

## Variants by feature type

Steps 1 to 4 never fixed the form of $p(x_j \mid y)$. Choosing it to match the feature type gives the named variants.

| Variant | Feature type | Model of p(x_j given y) | Typical use
| --- | --- | --- | --- |
| Gaussian | continuous | normal with per-class mean and variance | sensor or tabular features
| Multinomial | counts | word-frequency distribution over the vocabulary | text classification with counts or TF-IDF
| Bernoulli | binary | presence or absence per feature | short texts, binary flags
| Complement | counts | fitted on all other classes | imbalanced text classes

**Gaussian naive Bayes** fits a bell curve (mean $\mu_{kj}$, variance $\sigma^2_{kj}$) to each feature within each class. Picture a pump monitor: healthy pumps vibrate around 5.2 mm/s (standard deviation 0.8); faulty pumps around 7.4 mm/s (standard deviation 1.1); 20% of pumps are faulty.

~~~dg-fig 03.01-gauss.svg
Gaussian naive Bayes with one feature. Each class's bell curve is scaled by its prior (left); dividing by their sum gives the posterior (right). The rare faulty class only wins above about 6.8 mm/s, past the point where the two raw curves cross.
~~~

At 6.5 mm/s the faulty density (0.26) is actually *higher* than the healthy one (0.13), yet after multiplying by the priors the scores are $0.2 \times 0.26 = 0.052$ against $0.8 \times 0.13 = 0.107$, so $p(\text{faulty}) = 0.33$. The prior pushes the boundary toward the rare class.

With several features, Gaussian naive Bayes fits one bell curve per feature and multiplies them. Geometrically, each class becomes an ellipse whose axes are parallel to the feature axes, because independence means no tilt.

~~~dg-fig 03.01-ellipses.svg
Gaussian naive Bayes draws one axis-aligned ellipse per class (rings at 1 and 2 standard deviations). Left: the features really are independent, the ellipses match the data and the boundary is sensible. Right: the clouds are tilted along the diagonal, the ellipses cannot follow, and the boundary bends in the wrong place.
~~~

**Bernoulli naive Bayes** models presence or absence: $\prod_j \theta_{kj}^{x_j}(1 - \theta_{kj})^{1 - x_j}$. The $(1 - \theta_{kj})$ factor means it also **penalises absence**: a typical spam word *missing* from a message lowers the spam score, which multinomial naive Bayes ignores.

**Complement naive Bayes** estimates each class's profile from all the *other* classes' text, which is more stable when classes are imbalanced.

## Naive Bayes in code

First the whole algorithm from scratch, which shows it is counting plus one matrix product:

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

In practice you use scikit-learn. Here it reproduces the hand example, then runs on raw text and on a real tabular dataset:

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

X = np.array([[3, 2, 0, 1], [1, 0, 4, 3]])          # summed counts: spam row, ham row
nb = MultinomialNB(alpha=1.0, class_prior=[0.6, 0.4]).fit(X, ["spam", "ham"])
print(nb.classes_)                                   # ['ham' 'spam']  (sorted)
print(nb.predict_proba([[1, 1, 1, 0]]).round(3))     # [[0.42 0.58]]  matches the hand result

docs = ["win a free prize now", "free offer win cash", "meeting moved to monday",
        "quarterly report attached", "free lunch at the meeting", "claim your free win"]
labels = ["spam", "spam", "ham", "ham", "ham", "spam"]
clf = make_pipeline(CountVectorizer(), MultinomialNB(alpha=0.5)).fit(docs, labels)
print(clf.predict(["free cash prize", "report for the monday meeting"]))   # ['spam' 'ham']

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
Xb, yb = load_breast_cancer(return_X_y=True)          # 569 tumours, 30 continuous features
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, random_state=0, stratify=yb)
g = GaussianNB().fit(Xtr, ytr)
print(g.score(Xte, yte))                               # 0.924
```

Training is one pass over the data, $O(nd)$, works on sparse matrices, and `partial_fit` updates the counts batch by batch.

## Reading the result

Check a fitted naive Bayes model in this order.

**1. Accuracy against baselines**: the majority class and logistic regression on the same features. On the breast-cancer split above, Gaussian naive Bayes gets 92.4% test accuracy, a standardised logistic regression 95.9%.

**2. The features driving each class.** Sort words by $\log\theta_{\text{spam},j} - \log\theta_{\text{ham},j}$. A surprising word near the top (a sender's name, a date) usually means leakage.

**3. The probabilities, separately from the classes.** On the breast-cancer test set, 95% of its predicted probabilities are below 0.01 or above 0.99, and on its 13 mistakes its average confidence is 0.987. Logistic regression's 7 mistakes average 0.69. The log loss tells the same story: 1.09 for naive Bayes against 0.085 for logistic regression. The data explains why: the tumour's radius, perimeter and area are 99% correlated with each other, and naive Bayes counts that one piece of evidence three times.

## Why it works anyway, and what it costs

Classification only needs the right class to score highest. Correlated features push the score in the *same direction* more than once, which inflates its size but often leaves its sign intact.

The cost shows up in the probabilities. If one feature is duplicated, its log-likelihood ratio is added twice, which is the same as squaring its likelihood ratio.

~~~dg-fig 03.01-overconf.svg
One feature that separates two classes, copied 1, 2 and 5 times. The decision boundary stays at zero, so accuracy is unchanged, but each copy steepens the curve: at x = 1 the posterior goes from 0.88 to 0.98 to 0.99995.
~~~

So naive Bayes probabilities are typically overconfident and need calibration (03.04) before you use them, for example to set a threshold by expected cost.

Ng and Jordan compared naive Bayes with logistic regression, its discriminative counterpart. Naive Bayes approaches its best error with far fewer examples (on the order of $\log d$ rather than $d$), but logistic regression usually reaches a lower error once data is plentiful.

~~~dg-fig 03.01-learning.svg
Simulated learning curves on 46 binary features, one weak feature copied six times. With 10 examples naive Bayes has the lower error; past a few hundred, logistic regression pulls ahead because it can learn to discount the copies.
~~~

That is the generative trade: a stronger assumption buys sample efficiency at the price of bias. With a few hundred labelled documents naive Bayes is a strong baseline; with a million, logistic regression usually wins.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| A class is never predicted when a certain word appears | zero count, no smoothing | set alpha above 0; tune it on a log grid
| Probabilities nearly all 0 or 1 | correlated or duplicated features | calibrate (03.04); remove near-duplicate features
| Every score is -inf or NaN | multiplying probabilities instead of adding logs | work in log space; use predict_log_proba
| Rare class swamped by the prior | strong class imbalance | complement NB, adjust class_prior, or move the threshold
| Gaussian NB poor on skewed features | feature is far from a bell curve | log-transform the feature, or bin it and use a categorical model
| Accuracy plateaus as data grows | the independence bias dominates | switch to logistic regression or a tree ensemble on the same features
| Negative-value error in MultinomialNB | centred or standardised inputs | feed raw counts or TF-IDF; never standardise them

Tune $\alpha$ by cross-validation on a log grid (0.001 to 1). For large vocabularies, 0.01 to 0.1 often beats 1, which flattens the profiles.

## Going deeper

**Multinomial naive Bayes is a linear classifier.** For a document with word counts $x_j$, the log score is $\log\pi_k + \sum_j x_j\log\theta_{kj}$, where $\pi_k$ is the prior. That is a bias $\log\pi_k$ plus a weight $\log\theta_{kj}$ per word times its count. For two classes, the difference of scores is

$$\log\frac{p(\text{spam} \mid x)}{p(\text{ham} \mid x)} = \log\frac{\pi_{\text{s}}}{\pi_{\text{h}}} + \sum_j x_j \log\frac{\theta_{\text{s}j}}{\theta_{\text{h}j}},$$

which is exactly the form of logistic regression, $b + w^\top x$. Same hypothesis class, different fitting: naive Bayes sets each weight by counting in isolation, while logistic regression fits them jointly by maximising $p(y \mid x)$, so it can shrink a word whose information another word already carries. (The multinomial coefficient does not depend on the class, so it cancels.)

**Gaussian naive Bayes is QDA with diagonal covariances.** Each class is a Gaussian with a diagonal covariance matrix, which is the special case of quadratic discriminant analysis (03.03) where all off-diagonal entries are zero. The boundary is quadratic in general, and becomes linear if every class shares the same variances. scikit-learn adds a small `var_smoothing` (a fraction of the largest feature variance) to every variance so that a feature that is constant within a class cannot produce a zero variance and an infinite density.

**Smoothing is a Bayesian prior.** With a symmetric Dirichlet($\alpha$) prior on each class's word distribution, the posterior *mean* is exactly $(N_{kj} + \alpha)/(N_k + \alpha d)$: $\alpha$ counts imaginary observations, whose influence fades as real data grows.

**Sample complexity.** Naive Bayes needs only $O(\log d)$ examples to get close to its own best error because each parameter is a simple average that converges independently. Logistic regression needs $O(d)$, but its limiting error is never worse than naive Bayes' on the same features. If the features truly are conditionally independent, naive Bayes is Bayes-optimal and its probabilities are correct.

## Key terms

| Term | Meaning
| --- | --- |
| Generative classifier | Models how each class produces inputs, p(x given y), plus class frequencies, then applies Bayes' rule
| Discriminative classifier | Models p(y given x) or a decision boundary directly, such as logistic regression
| Prior | How common a class is before seeing the input
| Likelihood | How probable the observed input is under one class
| Posterior | Probability of a class after seeing the input
| Conditional independence | Given the class, knowing one feature tells you nothing more about another
| Additive (Laplace) smoothing | Adding alpha pseudo-counts so no estimated probability is exactly zero
| Underflow | A product of many small probabilities rounding to 0; avoided by summing logs
| Overconfidence | Predicted probabilities more extreme than the true frequencies of being right

<div class="take">
<div class="lab">Key takeaway</div>

naive Bayes is a generative classifier that trains by counting under a conditional-independence assumption and predicts by adding one log-probability per feature. It is fast, data-efficient and a strong text baseline, but smooth the counts, work in logs, and do not trust its probabilities without calibration.

</div>
