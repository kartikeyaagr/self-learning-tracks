---
id: "10.04"
course: "ml-maths-course"
module: "10"
slug: "from-counts-to-densities"
title: "From Counts to Densities"
keyIdeas: ["For a continuous quantity, any single exact value has probability 0; probability lives on intervals.", "A histogram with bar heights scaled so the total area is 1 approaches a density as the bins shrink.", "Density is probability per unit length, not a probability."]
question: "10,000 samples fall into bins of width 0.1. One bin holds 400 samples. What density does its bar estimate?"
answer: "Step 1, the fraction in the bin is ⟦3268⟧. Step 2, divide by the width: ⟦3269⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous random variables"]
tools: ["NumPy"]
topic: {"id": "10.02", "title": "Continuous random variables"}
---

## The big picture

A label is one of a few values, but a weight, a pixel intensity or a latency can be any real number. "The probability of exactly 0.31" is zero, because there are infinitely many possible values. Probability is spread out like a fluid, and the density says how thick it is at each point.

## Histograms become curves

Draw many samples, put them in bins, and divide each count by (total samples × bin width). The bars then have total area 1. With more samples and narrower bins, the tops of the bars trace a smooth curve: the **density**.

```
rng = np.random.default_rng(0)
x = rng.normal(size=100_000)
h, edges = np.histogram(x, bins=40, range=(-4, 4), density=True)
print(round((h * np.diff(edges)).sum(), 3), round(h.max(), 2))  # 1.0 0.4
```

The tallest bar is about 0.4, close to the standard normal's peak <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><mrow><mn>2</mn><mi>π</mi></mrow></msqrt><mo>≈</mo><mn>0.399</mn></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=Math.exp(-x*x/2)/Math.sqrt(2*Math.PI) x=-4,4 y=0,0.45 tk=1 dom=-1,1,cc | The standard normal density. The shaded strip from −1 to 1 holds about 68% of the probability; the curve's height is not a probability.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**probability of a point**

The probability that a standard normal is in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0.31</mn><mo separator="true">,</mo><mn>0.31</mn><mo>+</mo><mi>h</mi><mo stretchy="false">]</mo></mrow></math></span> is about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.38</mn><mi>h</mi></mrow></math></span>. As <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi><mo>→</mo><mn>0</mn></mrow></math></span> it goes to 0.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

probabilities are areas under the density; a single point has none.

</div>

## Where this breaks in practice

- Histogram shape depends on bin width. Too wide hides structure, too narrow shows noise.
