---
id: "10.30"
course: "ml-maths-course"
module: "10"
slug: "the-poisson-as-a-limit-of-the-binomial"
title: "The Poisson as a Limit of the Binomial"
keyIdeas: ["Binomial(⟦3493⟧) with ⟦233⟧ large, ⟦102⟧ small and ⟦3515⟧ fixed is close to Poisson(⟦559⟧).", "When ⟦3496⟧ is large (above about 10), the binomial is also close to a Gaussian.", "The approximations replace an awkward sum with a simpler formula."]
question: "1000 requests with error rate 1%. Compare the binomial and Poisson probabilities of zero errors."
answer: "Step 1, binomial: ⟦3517⟧. Step 2, Poisson(10): ⟦3518⟧. Close."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

Many chances for a rare event: the binomial becomes the Poisson. This is why the Poisson fits so many counts.

## How they connect

~~~dg-flow One coin flip grows into the family.
Bernoulli(p) -(sum n independent)-> Binomial(n, p) -(n large, p small, np fixed)-> *Poisson(np)
Binomial(n, p) -(np(1-p) large)-> approximately Gaussian
~~~

```
from scipy import stats
print(round(stats.binom(1000, 0.01).pmf(0) * 1e5, 2), round(stats.poisson(10).pmf(0) * 1e5, 2))  # 4.32 4.54
k = np.arange(30)
print(round(np.abs(stats.binom(1000, 0.01).pmf(k) - stats.poisson(10).pmf(k)).max(), 4))  # 0.0006
```

The largest gap between the two PMFs is under 0.001.

<div class="take">
<div class="lab">Key takeaway</div>

many trials with a small success rate give a Poisson count with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>=</mo><mi>n</mi><mi>p</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- The approximation is poor when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is not small: Binomial(20, 0.5) is not Poisson-shaped.
