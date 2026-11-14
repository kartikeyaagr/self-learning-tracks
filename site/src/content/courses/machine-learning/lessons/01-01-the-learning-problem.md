---
id: "01.01"
course: "machine-learning"
module: "01"
slug: "the-learning-problem"
title: "The learning problem"
keyIdeas: ["Supervised learning assumes examples $(x, y)$ are drawn from an unknown distribution $\\mathcal{D}$; the goal is a function $h$ that predicts $y$ well on *new* draws, not on the ones you have.", "You pick a **hypothesis class** $\\mathcal{H}$ (lines, trees, networks) and a **loss** $\\ell(h(x), y)$; together they define what \"good\" means and what is even reachable.", "**Empirical risk minimisation** picks $\\hat h = \\arg\\min_{h \\in \\mathcal{H}} \\frac1n \\sum_i \\ell(h(x_i), y_i)$, a stand-in for the true risk $R(h) = \\mathbb{E}_{\\mathcal{D}}[\\ell(h(x), y)]$ you cannot compute.", "The loss you train on is often a smooth surrogate (log loss, hinge) for the one you care about (0-1 error), because the true one has no useful gradient.", "Every later lesson changes one of three knobs: the class, the loss, or how the minimisation is constrained."]
question: "A team trains a fraud model by minimising log loss, reports 0.3% training error, and ships. On live traffic the model flags almost nothing. Name two distinct places in the learning setup where this could have gone wrong, and how you would tell them apart."
answer: "One is the **generalisation gap**: the class is flexible enough to memorise the training set, so low empirical risk says little about true risk. You would see it by evaluating on a held-out split drawn from the same period: if held-out error is far above 0.3%, the model overfit. The other is a violation of the **i.i.d. assumption**: live traffic comes from a different distribution (new fraud patterns, a different country mix, a feature computed differently in production). Then held-out error from the old data can look fine while live performance collapses; comparing feature distributions between training and live data (or evaluating on a later time slice) exposes it. A third suspect is the **loss/metric mismatch**: with 0.3% fraud, predicting \"not fraud\" always already gives 0.3% error, so the reported number may simply be the base rate."
sources: [{"title": "Understanding Machine Learning (Shalev-Shwartz and Ben-David)", "url": "https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/", "note": "Free PDF. Chapters 2-3 define the learning model, ERM and PAC learning precisely."}, {"title": "Cornell CS4780: Machine learning setup", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote01_MLsetup.html", "note": "Short notes on the D, H, loss framing with good intuition for why memorisation fails."}, {"title": "Cornell CS4780: Empirical risk minimisation", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote10.html", "note": "Surrogate losses (hinge, log, exponential) side by side and how they bound 0-1 loss."}]
practice: ["linear-regression", "loss-functions", "log-loss-binary-cross-entropy", "regression-metrics"]
tags: ["empirical risk", "loss functions"]
tools: ["NumPy"]
---

## The setup

Supervised learning has four ingredients. An **input** $x$, usually a vector of $d$ features ($x \in \mathbb{R}^d$). A **label** $y$: a real number for regression, one of $K$ classes for classification. An unknown **data distribution** $\mathcal{D}$ over pairs $(x, y)$. And a **training set** of $n$ pairs drawn independently from it, $S = \{(x_1, y_1), \dots, (x_n, y_n)\}$.

The job is to output a function $h: x \mapsto \hat y$ that does well on *future* draws from $\mathcal{D}$. That one word, future, is what makes machine learning different from curve fitting. A lookup table that memorises $S$ fits the training set perfectly and is useless on the next example.

Two assumptions hide in this setup and are worth saying out loud:

- **i.i.d. sampling**: training and future examples come from the same $\mathcal{D}$, independently. Most real failures (drift, leakage, selection bias) are violations of this.
- **Labels are noisy**: even the best possible predictor makes mistakes, because $y$ is not a deterministic function of $x$. The error of that best predictor is the *Bayes error*, a floor no model can beat.

~~~dg-flow The learning loop: you choose the class and the loss; the data picks the function inside the class.
Sample S from D -(choose)-> Hypothesis class H -(choose)-> Loss -(minimise on S)-> *Learned h -(evaluate on new data)-> Test risk
~~~

## Hypothesis classes and losses

The **hypothesis class** $\mathcal{H}$ is the set of functions you are willing to consider. Linear models $h_w(x) = w^\top x + b$ form a class with $d + 1$ parameters. Decision trees of depth 5 are another class. A ResNet-50 is a class with 25 million parameters. Picking the class is the single biggest modelling decision, because ERM can only return something inside it.

The **loss** $\ell(\hat y, y)$ scores one prediction. Common choices:

| Task | Loss | Formula |
| --- | --- | --- |
| Regression | squared error | $(\hat y - y)^2$ |
| Regression, robust | absolute / Huber | $|\hat y - y|$, quadratic near 0 |
| Binary classification | 0-1 | $\mathbb{1}[\hat y \ne y]$ |
| Binary, probabilistic | log loss | $-y\log p - (1-y)\log(1-p)$ |
| Binary, margin | hinge | $\max(0, 1 - y\,f(x))$ with $y \in \{-1, 1\}$ |

The loss encodes what errors cost. Squared error punishes one prediction that is off by 10 as much as a hundred that are off by 1, so outliers dominate. Absolute error does not. If the business cost of a false negative is ten times a false positive, the loss should say so.

## Empirical risk minimisation

The quantity you care about is the **true risk** $R(h) = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(h(x), y)]$. You cannot compute it, because $\mathcal{D}$ is unknown. You can compute the **empirical risk** on the sample:

$$\hat R_S(h) = \frac{1}{n}\sum_{i=1}^{n} \ell(h(x_i), y_i)$$

ERM picks the hypothesis with the smallest empirical risk, $\hat h = \arg\min_{h\in\mathcal{H}} \hat R_S(h)$. For a fixed $h$, $\hat R_S(h)$ is an unbiased estimate of $R(h)$. The catch is that $\hat h$ is *chosen using* $S$, so $\hat R_S(\hat h)$ is biased low: you picked the function that happens to look best on this sample. The gap $R(\hat h) - \hat R_S(\hat h)$ is the **generalisation gap**, and the next lesson is about what controls it.

```python
import numpy as np
# ERM for a 1-D linear class under squared loss, by brute force over a grid of slopes
x = np.array([1., 2., 3., 4.]); y = np.array([2.1, 3.9, 6.2, 7.8])
slopes = np.linspace(0, 4, 401)
risk = [np.mean((w * x - y) ** 2) for w in slopes]
w_hat = slopes[int(np.argmin(risk))]     # about 1.99
```

Real ERM is never a grid search; it is a closed form (least squares) or an iterative optimiser (gradient descent). But the object being minimised is the same average.

## Surrogate losses

The loss you care about and the loss you optimise are often different. Minimising 0-1 error over linear classifiers is NP-hard in general, and its gradient is zero almost everywhere, so gradient methods get no signal. Instead you minimise a **convex surrogate** that upper-bounds it: log loss gives logistic regression, hinge gives the SVM, both covered in Module 02. You still *report* the metric you care about (accuracy, F1, cost), measured on held-out data.

<div class="ex">
<div class="lab">Worked example</div>

**two classes, one sample, three hypotheses**

Data: 100 emails, 20 spam. Class $\mathcal{H}_0$ = constant predictors; $\mathcal{H}_1$ = linear models on word counts; $\mathcal{H}_2$ = a lookup table keyed on the exact email text.

ERM in $\mathcal{H}_0$ predicts "not spam" for everything: empirical 0-1 risk 0.20. ERM in $\mathcal{H}_1$ might reach 0.04. ERM in $\mathcal{H}_2$ reaches 0.00.

On new mail, $\mathcal{H}_2$ has never seen any of the texts and falls back to a default guess, so its true risk is about 0.20 again. Lowest training risk is not lowest true risk; the class decides how much the training number can be trusted.

</div>

## Where this course goes next

Every method in this course is an instance of the same template: a class, a loss, and a way of minimising empirical risk, often with a constraint added. Linear regression is squared loss over linear functions. Logistic regression is log loss over the same class. An SVM is hinge loss plus a norm penalty. k-NN and naive Bayes reach the template from different directions (memorisation and probabilistic modelling), and Module 03 shows how. The maths behind least squares and likelihood is covered in depth in the ML Maths course (Applications and Inference modules); here the focus is on how the pieces are chosen and how they fail.

<div class="take">
<div class="lab">Key takeaway</div>

learning is choosing a class and a loss, then minimising the average loss on a sample as a proxy for the expected loss on new data. The proxy is biased toward the sample, and controlling that bias is the rest of the subject.

</div>
