---
id: "10.07"
course: "ml-maths-course"
module: "10"
slug: "quantiles-and-percentiles"
title: "Quantiles and Percentiles"
keyIdeas: ["The quantile function ⟦3295⟧ returns the value below which a fraction ⟦280⟧ of the probability lies.", "The median is ⟦3296⟧; \"p99 latency\" is ⟦3297⟧.", "For a long right tail, the median is below the mean."]
question: "Latency is exponential with mean 200 ms. Find the median and the p99."
answer: "Step 1, ⟦3299⟧. Step 2, median: ⟦3300⟧, so ⟦3301⟧ ms. Step 3, p99: ⟦3302⟧, so ⟦3303⟧ ms."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous random variables"]
tools: ["SciPy"]
topic: {"id": "10.02", "title": "Continuous random variables"}
---

## The big picture

Engineers report p50, p95 and p99 latency, and confidence intervals are built from quantiles. A quantile is the CDF read backwards.

## Reading the CDF backwards

Pick a probability <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span> on the vertical axis, go across to the CDF, and read off <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> below.

~~~dg-raw
%%viz f_curve f=1-Math.exp(-x/200) x=0,1000 y=0,1 tk=200 ytk=0.25 pts=138.6,median_139;921,p99_921 | The CDF of an exponential latency with mean 200 ms. The median is at 139 ms, the p99 at 921 ms.
~~~

```
from scipy import stats
d = stats.expon(scale=200)
print(round(d.median(), 1), round(d.ppf(0.99), 1))  # 138.6 921.0
```

`ppf` ("percent point function") is SciPy's name for the quantile function.

<div class="ex">
<div class="lab">Worked example</div>

**the standard normal**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">Φ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mn>0.975</mn><mo stretchy="false">)</mo><mo>≈</mo><mn>1.96</mn></mrow></math></span>. That is where "± 1.96 standard errors" in a 95% interval comes from.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a quantile is the value with a given fraction of probability below it.

</div>

## Where this breaks in practice

- Sample percentiles from few data points are noisy, especially in the tail. A p99 from 200 requests rests on two of them.
