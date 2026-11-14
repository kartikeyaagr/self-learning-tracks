---
id: "10.34"
course: "ml-maths-course"
module: "10"
slug: "overdispersion-and-the-negative-binomial"
title: "Overdispersion and the Negative Binomial"
keyIdeas: ["Real counts often have variance much larger than their mean: overdispersion.", "A Poisson model then underestimates uncertainty and treats normal swings as anomalies.", "The negative binomial adds a dispersion parameter: variance ⟦3545⟧."]
question: "Hourly errors have mean 3 and variance 12. Is a Poisson model reasonable?"
answer: "Step 1, a Poisson has variance equal to its mean, 3. Step 2, the data's variance is 4 times that, so the counts are overdispersed and a negative binomial fits better."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

The Poisson assumes events are independent with a constant rate. When the rate itself varies (busy hours, bursty users), counts spread out more.

## A Poisson with a varying rate

If the rate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span> is itself random (Gamma distributed), the resulting count is negative binomial. Its variance is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mo>+</mo><msup><mi>μ</mi><mn>2</mn></msup><mi mathvariant="normal">/</mi><mi>r</mi></mrow></math></span>, where smaller <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> means more spread.

```
from scipy import stats
rng = np.random.default_rng(0)
lam = rng.gamma(shape=1, scale=3, size=500000)
k = rng.poisson(lam)
print(round(k.mean(), 1), round(k.var(), 1))  # 3.0 12.0
r = 1; p = r / (r + 3)
print(stats.nbinom(r, p).mean(), stats.nbinom(r, p).var())  # 3.0 12.0
```

~~~dg-raw
%%viz f_bars v=5,25,14.9,18.8,22.4,14.1,22.4,10.5 l=P0,NB0,P1,NB1,P2,NB2,P3,NB3 hi=none yl=%_chance | Poisson(3) (P) against a negative binomial with the same mean 3 and variance 12 (NB). The negative binomial puts much more weight on 0 and on large counts.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

if count variance exceeds the mean, use a negative binomial, not a Poisson.

</div>

## Where this breaks in practice

- Check the variance-to-mean ratio before fitting any count model.
