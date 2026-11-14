---
id: "10.36"
course: "ml-maths-course"
module: "10"
slug: "the-exponential-distribution"
title: "The Exponential Distribution"
keyIdeas: ["Exponential(⟦559⟧): density ⟦3557⟧ for ⟦2982⟧, mean ⟦3558⟧, variance ⟦3559⟧.", "It is memoryless: ⟦3560⟧.", "It describes the gap between events of a Poisson process."]
question: "Waiting time has mean 10 minutes. You've waited 10 already. What is the chance of waiting at least 5 more?"
answer: "Step 1, memorylessness: it is ⟦3562⟧. Step 2, ⟦3563⟧, the same as from the start."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.06", "title": "Continuous probability distributions"}
---

## The big picture

Times between independent random events, such as request arrivals, are exponential. It is the continuous cousin of the geometric.

## The density

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>λ</mi><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>x</mi></mrow></msup><mo separator="true">,</mo><mspace width="1em"/><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mo>−</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>x</mi></mrow></msup></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=0.1*Math.exp(-0.1*x) x=0,40 y=0,0.11 tk=10 pts=10,mean_10 | Exponential with mean 10 minutes. Short waits are most likely, but long ones are not rare.
~~~

```
from scipy import stats
d = stats.expon(scale=10)
print(round(d.sf(15) / d.sf(10), 4), round(d.sf(5), 4))  # 0.6065 0.6065
```

<div class="take">
<div class="lab">Key takeaway</div>

the exponential models waiting times and forgets how long you've already waited.

</div>

## Where this breaks in practice

- Real latencies are rarely memoryless: a request that has been slow is often stuck, so p99 latency is usually worse than an exponential model predicts.
