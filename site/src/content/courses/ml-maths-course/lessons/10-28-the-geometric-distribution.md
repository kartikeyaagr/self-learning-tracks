---
id: "10.28"
course: "ml-maths-course"
module: "10"
slug: "the-geometric-distribution"
title: "The Geometric Distribution"
keyIdeas: ["Geometric(⟦102⟧) is the number of trials up to and including the first success: ⟦3503⟧.", "Mean ⟦3504⟧, variance ⟦3505⟧.", "It is memoryless: past failures do not change the chance of the next trial."]
question: "Each training run succeeds with probability 0.25. What is the chance the first success is run 3, and the expected number of runs?"
answer: "Step 1, ⟦3508⟧. Step 2, mean ⟦3509⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

"How long until it works?" Retries, rejection sampling and random restarts all follow a geometric distribution.

## The PMF

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>K</mi><mo>=</mo><mi>k</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>p</mi><msup><mo stretchy="false">)</mo><mrow><mi>k</mi><mo>−</mo><mn>1</mn></mrow></msup><mi>p</mi></mrow></math></span></span>
<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>−</mo><mn>1</mn></mrow></math></span> failures, then a success.

~~~dg-raw
%%viz f_bars v=25,18.8,14.1,10.5,7.9,5.9,4.4,3.3 l=1,2,3,4,5,6,7,8 hi=none yl=%_chance_first_success | Geometric(0.25). Each bar is 0.75 times the one before. The mean is 4 runs.
~~~

```
from scipy import stats
d = stats.geom(0.25)
print(round(d.pmf(3), 4), d.mean())  # 0.1406 4.0
```

<div class="ex">
<div class="lab">Worked example</div>

**rejection sampling**

If a proposal is accepted with probability 0.1, you need on average 10 proposals per sample.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

the geometric counts trials to the first success, with mean <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mi>p</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- SciPy's `geom` counts trials including the success; some texts count only the failures, shifting the mean by 1.
