---
id: "12.06"
course: "ml-maths-course"
module: "12"
slug: "the-central-limit-theorem"
title: "The Central Limit Theorem"
keyIdeas: ["With finite variance, ⟦3867⟧ is approximately ⟦3868⟧ for large ⟦233⟧, whatever the original distribution's shape.", "It is why the Gaussian appears whenever many small independent effects add up.", "Skewed data need larger ⟦233⟧ before the approximation is good."]
question: "⟦3864⟧, ⟦3865⟧. Use the CLT to estimate ⟦3866⟧."
answer: "Step 1, ⟦3869⟧, standard deviation 0.1. Step 2, 1.2 is 2 standard deviations up: ⟦3870⟧. The exact answer is about 0.028."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["SciPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

The law of large numbers says the average converges. The CLT says what its error looks like on the way: a bell curve.

## A skewed start, a Gaussian average

~~~dg-raw
%%viz f_curve f=Math.exp(-x) g=Math.exp(-(x-1)*(x-1)/0.02)/Math.sqrt(0.02*Math.PI)/4 x=0,3 y=0,1.05 tk=0.5 fl=one_exponential_value gl=average_of_100_(height_÷_4) | One Exponential(1) value is strongly skewed (green). The average of 100 of them is close to a narrow bell curve around 1 (orange, drawn at a quarter of its height).
~~~

```
from scipy import stats
print(round(stats.norm.sf(2), 3), round(stats.gamma(100, scale=1 / 100).sf(1.2), 3))  # 0.023 0.028
```

The sum of exponentials has an exact Gamma distribution; the CLT's answer is close but slightly off, because some skew remains at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>100</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

averages of many independent pieces are approximately Gaussian, with standard deviation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mi mathvariant="normal">/</mi><msqrt><mi>n</mi></msqrt></mrow></math></span>.

</div>

## Where this breaks in practice

- Heavy tails slow the CLT down a lot; with infinite variance it does not apply.
