---
id: "15.04"
course: "ml-maths-course"
module: "15"
slug: "mutual-information-and-information-gain"
title: "Mutual information and information gain"
keyIdeas: ["Mutual information $I(X;Y) = H(Y) - H(Y\\mid X)$ is the reduction in uncertainty about $Y$ from knowing $X$; it is symmetric, non-negative and zero only for independent variables.", "Equivalently, $I(X;Y) = D_{\\mathrm{KL}}(p(x,y)\\,\\|\\,p(x)p(y))$: how far the joint distribution is from independence.", "A decision tree's information gain is the mutual information between a split and the label, measured on the examples at that node.", "Mutual information detects any dependence, including non-linear ones that correlation misses, and processing can only lose it (the data processing inequality)."]
question: "A node in a decision tree holds 10 positive and 10 negative examples. Split A sends 8 positives and 2 negatives left, and the rest right. Split B sends 10 examples (5 positive, 5 negative) left and the other 10 (5 and 5) right. (a) Compute the information gain of each split in bits. (b) A third split uses a customer-ID feature with a different value for every example. What is its information gain, and why is it a bad split?"
answer: "(a) Parent entropy is 1 bit. Split A: each child has a 0.8/0.2 mix, with entropy $-0.8\\log_2 0.8 - 0.2\\log_2 0.2 \\approx 0.722$ bits, so the weighted child entropy is 0.722 and the gain is about 0.278 bits. Split B: both children are 50/50, entropy 1 bit, gain 0. (b) Every child holds one example and has entropy 0, so the gain is the full 1 bit, the maximum. But the split memorises the training examples and says nothing about new customers. Information gain favours features with many values; the gain ratio (gain divided by the split's own entropy, here $\\log_2 20 \\approx 4.32$ bits) or a minimum leaf size counters this."
sources: [{"title": "Wikipedia, Mutual information", "url": "https://en.wikipedia.org/wiki/Mutual_information", "note": "Definitions, identities and properties."}, {"title": "Wikipedia, Information gain (decision tree)", "url": "https://en.wikipedia.org/wiki/Information_gain_(decision_tree)", "note": "Information gain as a split criterion."}, {"title": "Kraskov, Stoegbauer and Grassberger, Estimating Mutual Information (2003)", "url": "https://arxiv.org/abs/cond-mat/0305641", "note": "The nearest-neighbour estimator for continuous variables."}, {"title": "MacKay, Information Theory, Inference, and Learning Algorithms (free book)", "url": "https://www.inference.org.uk/itprnn/book.pdf", "note": "Chapter 8 on dependent random variables."}]
practice: ["decision-tree-classifier-cart", "random-forest", "independence-testing", "chi-square-goodness-of-fit"]
tags: ["mutual information", "information gain", "feature selection", "data processing inequality"]
tools: ["NumPy", "scikit-learn"]
---

## Definition

Mutual information measures how much knowing one variable tells you about another:
$I(X;Y) = H(Y) - H(Y \mid X) = H(X) - H(X \mid Y) = H(X) + H(Y) - H(X, Y)$.
It is symmetric: $X$ tells you as much about $Y$ as $Y$ tells you about $X$. It also equals the KL divergence between the joint distribution and the product of the marginals,
$I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$,
so it is never negative and is zero exactly when $X$ and $Y$ are independent.

~~~dg-flow The pieces of joint uncertainty.
H(X, Y) -(splits into)-> H(X given Y) -> *I(X; Y) -> H(Y given X)
~~~

The joint entropy is the part only in $X$, plus the shared part, plus the part only in $Y$. The shared part is mutual information.

## Information gain in decision trees

At a node, the label $Y$ has some entropy. A candidate split $S$ divides the examples into children; the weighted average of the children's label entropies is $H(Y\mid S)$. The information gain is
$\mathrm{IG} = H(Y) - H(Y\mid S) = I(S; Y)$,
computed on the examples at that node. The tree picks the split with the largest gain. The Gini impurity, used by CART by default, is a close cousin that avoids logarithms and usually picks similar splits.

Information gain favours features with many distinct values, because splitting into many small groups makes each group purer by chance. Remedies are the gain ratio (divide by the entropy of the split itself), limits on leaf size, and evaluating splits on held-out data.

## Beyond correlation

Correlation measures linear dependence only. Let $X$ be uniform on $\{-1, 0, 1\}$ and $Y = X^2$. The correlation is 0, but $Y$ is a function of $X$, so $I(X;Y) = H(Y) \approx 0.918$ bits. Mutual information catches any kind of dependence, which makes it useful for:

- Feature screening: rank features by their estimated mutual information with the target.
- Detecting redundancy: two features with high mutual information carry overlapping signals.
- Understanding representations: how much information about the input or label a layer keeps.

<div class="ex">
<div class="lab">Worked example</div>

**a useful feature that correlation missed**

In a churn model, the correlation between days since last login and churn was weak, so the feature was dropped. A plot showed the reason: both very short and very long gaps predicted low churn (the long gaps were mostly seasonal users who came back each year), while medium gaps predicted high churn. Mutual information ranked the feature among the top five, and adding it back with a tree model improved validation log loss noticeably.

</div>

## The data processing inequality

If $X \to Y \to Z$ form a chain (so $Z$ depends on $X$ only through $Y$), then $I(X;Z) \le I(X;Y)$. No processing of $Y$, however clever, can create information about $X$ that $Y$ did not have. In ML terms: a network's deeper layers cannot know more about the input than earlier layers did, and a model built on a lossy feature pipeline is limited by what the pipeline kept. It is a useful sanity check when someone claims a post-processing step "adds information".

## Conditional mutual information

$I(X;Y\mid Z)$ measures the dependence between $X$ and $Y$ that remains once $Z$ is known. A feature that seems informative may carry nothing new once a stronger, correlated feature is included: its conditional mutual information with the label, given that feature, is small. Greedy feature selection methods use this to avoid picking redundant features.

## Estimating mutual information

For discrete variables with plenty of data, plug in the empirical joint frequencies. With few samples per cell, the plug-in estimate is biased upwards: independent variables show positive mutual information by chance. For continuous variables, binning makes the answer depend on the bin width; nearest-neighbour estimators (Kraskov and colleagues) avoid binning and work well in low dimensions. Lesson 15.06 discusses estimation further.

## Common mistakes

- Comparing mutual information values across targets with different entropies; normalise (for example by $\min(H(X), H(Y))$) when you need a 0 to 1 scale.
- Trusting high mutual information for features with many rare values; check against a shuffled label (lesson 15.06).
- Concluding causation: mutual information, like correlation, is symmetric and says nothing about direction.

<div class="take">
<div class="lab">Key takeaway</div>

mutual information is the shared uncertainty between two variables, equal to the KL divergence from independence; it is the information gain behind decision trees, detects non-linear dependence, and cannot be increased by processing.

</div>
