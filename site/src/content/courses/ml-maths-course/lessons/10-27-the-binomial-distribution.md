---
id: "10.27"
course: "ml-maths-course"
module: "10"
slug: "the-binomial-distribution"
title: "The Binomial Distribution"
keyIdeas: ["Binomial(⟦3493⟧) counts successes in ⟦233⟧ independent Bernoulli(⟦102⟧) trials: ⟦3494⟧.", "Mean ⟦3495⟧ and variance ⟦3496⟧, from linearity and adding independent variances.", "Accuracy on a test set of ⟦233⟧ independent examples is a binomial count divided by ⟦233⟧."]
question: "5 emails, each spam with probability 0.3. What is the probability exactly 2 are spam?"
answer: "Step 1, ⟦3500⟧ ways. Step 2, each has probability ⟦3501⟧. Step 3, total ⟦3502⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

"How many of these <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> things happen?" is a binomial question when the things are independent with the same probability: errors in a batch, clicks among impressions.

## The PMF

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>K</mi><mo>=</mo><mi>k</mi><mo stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true">(</mo><mfrac linethickness="0px"><mi>n</mi><mi>k</mi></mfrac><mo fence="true">)</mo></mrow><msup><mi>p</mi><mi>k</mi></msup><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>p</mi><msup><mo stretchy="false">)</mo><mrow><mi>n</mi><mo>−</mo><mi>k</mi></mrow></msup></mrow></math></span></span>

~~~dg-raw
%%viz f_bars v=16.8,36,30.9,13.2,2.8,0.2 l=0,1,2,3,4,5 hi=none yl=%_chance | Binomial(5, 0.3): the chance of k spam emails out of 5. The mean, 1.5, falls between the two tallest bars.
~~~

```
from scipy import stats
d = stats.binom(5, 0.3)
print(round(d.pmf(2), 4), d.mean(), round(d.var(), 2))  # 0.3087 1.5 1.05
```

<div class="ex">
<div class="lab">Worked example</div>

**errors in 1000 requests**

Error rate 1%. Mean <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1000</mn><mo>×</mo><mn>0.01</mn><mo>=</mo><mn>10</mn></mrow></math></span>, standard deviation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mn>9.9</mn></msqrt><mo>≈</mo><mn>3.15</mn></mrow></math></span>.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

the binomial counts successes in independent trials; mean <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mi>p</mi></mrow></math></span>, variance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mi>p</mi><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>p</mi><mo stretchy="false">)</mo></mrow></math></span>.

</div>

## Where this breaks in practice

- Errors that cluster (one bad upstream service fails many requests) break independence and make counts more variable than the binomial predicts.
