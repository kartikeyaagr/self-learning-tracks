---
id: "12.02"
course: "ml-maths-course"
module: "12"
slug: "estimators-and-sampling-distributions"
title: "Estimators and Sampling Distributions"
keyIdeas: ["An estimator ⟦3847⟧ is any recipe that turns a sample into a guess.", "Because the sample is random, ⟦3848⟧ is a random variable; its distribution is the sampling distribution.", "Different estimators of the same quantity have different sampling distributions."]
question: "Which is the better estimator of a normal population's centre from 11 points: the mean or the median?"
answer: "Step 1, both are centred on ⟦1238⟧. Step 2, for normal data the median's variance is about ⟦3850⟧ times the mean's, so the mean is better."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["NumPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

You never see the population, only one sample. To judge a method, imagine running it on many samples and look at the spread of answers.

## The sampling distribution

~~~dg-flow From one sample to a sampling distribution.
Population -(draw n points)-> sample -(apply the estimator)-> one estimate -(repeat many times)-> *sampling distribution of the estimator
~~~

```
rng = np.random.default_rng(0)
x = rng.normal(0, 1, size=(200000, 11))
print(round(x.mean(1).var(), 3), round(np.median(x, 1).var(), 3))  # 0.091 0.137
```

The mean's variance is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>11</mn><mo>≈</mo><mn>0.091</mn></mrow></math></span>. The median's is about 1.5 times larger.

~~~dg-raw
%%viz f_bars v=0.091,0.137 l=mean,median hi=none yl=variance_of_estimate_(n_=_11) | Two estimators of the same centre. For normal data the mean scatters less, so it is the better choice; for heavy-tailed data the order flips.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

an estimator is judged by its sampling distribution: where it centres and how much it spreads.

</div>

## Where this breaks in practice

- With outliers the median wins: one wild value can move the mean arbitrarily far.
