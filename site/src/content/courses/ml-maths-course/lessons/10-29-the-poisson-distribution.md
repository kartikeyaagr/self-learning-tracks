---
id: "10.29"
course: "ml-maths-course"
module: "10"
slug: "the-poisson-distribution"
title: "The Poisson Distribution"
keyIdeas: ["Poisson(⟦559⟧) models counts of events in a fixed interval: ⟦3510⟧.", "Its mean and variance are both ⟦559⟧.", "Requests per second, typos per page and failures per hour are classic examples."]
question: "A service averages 3 failed jobs per hour. What is the chance of no failures in an hour, and of 8 or more?"
answer: "Step 1, ⟦3513⟧. Step 2, ⟦3514⟧, about once every 84 hours by chance."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

Counts with no fixed upper limit, produced by many independent rare events, are modelled with the Poisson.

## The PMF

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>K</mi><mo>=</mo><mi>k</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi></mrow></msup><msup><mi>λ</mi><mi>k</mi></msup></mrow><mrow><mi>k</mi><mo stretchy="false">!</mo></mrow></mfrac></mrow></math></span></span>

~~~dg-raw
%%viz f_bars v=5,14.9,22.4,22.4,16.8,10.1,5,2.2,0.8 l=0,1,2,3,4,5,6,7,8 hi=none yl=%_chance | Poisson(3): the chance of k failures in an hour. 2 and 3 are equally likely; 8 or more happens about 1.2% of the time.
~~~

```
from scipy import stats
d = stats.poisson(3)
print(round(d.pmf(0), 3), round(d.sf(7), 4))  # 0.05 0.0119
```

`sf(7)` is the survival function, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>K</mi><mo>&gt;</mo><mn>7</mn><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>K</mi><mo>≥</mo><mn>8</mn><mo stretchy="false">)</mo></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

the Poisson counts rare independent events; its mean equals its variance.

</div>

## Where this breaks in practice

- A single bad hour with 8 failures is weak evidence of a problem. Look for sustained shifts.
