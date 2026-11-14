---
id: "12.05"
course: "ml-maths-course"
module: "12"
slug: "the-law-of-large-numbers"
title: "The Law of Large Numbers"
keyIdeas: ["As ⟦233⟧ grows, ⟦3863⟧: averages settle down to the true mean.", "It needs a finite mean; the Cauchy distribution has none and its averages never settle.", "It justifies Monte Carlo estimates and training on sample averages."]
question: "A running average of die rolls is at 3.9 after 10 rolls. Do later rolls \"make up\" for the high start?"
answer: "Step 1, no: each roll is independent with mean 3.5. Step 2, the early excess is not cancelled but diluted, since it is divided by an ever larger ⟦233⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["NumPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

Why does averaging work at all? Because the noise in an average shrinks to zero as the sample grows.

## Averages settle down

~~~dg-raw
%%viz f_curve f=3.5+3.42/Math.sqrt(x) g=3.5-3.42/Math.sqrt(x) x=1,500 y=1,6 tk=100 fl=3.5_+_2_SE gl=3.5_−_2_SE | The band where the running average of die rolls usually sits after n rolls (±2 standard errors around 3.5). It narrows toward the true mean.
~~~

```
rng = np.random.default_rng(0)
r = rng.integers(1, 7, 100000)
run = np.cumsum(r) / np.arange(1, r.size + 1)
print([float(round(run[n - 1], 2)) for n in (10, 1000, 100000)])  # [2.8, 3.6, 3.5]
c = np.cumsum(rng.standard_cauchy(100000)) / np.arange(1, 100001)
print(abs(c[-1] - c[49999]) > 0.01)  # True
```

The die's running average converges; the Cauchy's keeps jumping.

<div class="take">
<div class="lab">Key takeaway</div>

with a finite mean, averages converge to it.

</div>

## Where this breaks in practice

- Convergence can be slow with heavy tails, even when the mean exists.
