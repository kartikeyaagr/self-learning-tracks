---
id: "12.10"
course: "ml-maths-course"
module: "12"
slug: "the-likelihood-function"
title: "The Likelihood Function"
keyIdeas: ["The likelihood ⟦3891⟧ is the data's probability viewed as a function of ⟦875⟧.", "Work with ⟦3892⟧: sums are stable and differentiate easily.", "The likelihood is not a distribution over ⟦875⟧; it need not integrate to 1."]
question: "Coin data H, H, T, H. Compare ⟦3889⟧ and ⟦3890⟧."
answer: "Step 1, ⟦3895⟧. Step 2, ⟦3896⟧. The data favour 0.75."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Likelihood-based estimation"]
tools: ["NumPy"]
topic: {"id": "12.02", "title": "Likelihood-based estimation"}
---

## The big picture

A model with unknown parameters assigns a probability to the data you saw. Reading that as a function of the parameters tells you which settings fit.

## Fixed data, varying parameter

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>L</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo>=</mo><munderover><mo>∏</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>p</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>∣</mo><mi>θ</mi><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">ℓ</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>log</mi><mo>⁡</mo><mi>p</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>∣</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=x*x*x*(1-x)/0.10547 x=0,1 y=0,1.1 tk=0.25 pts=0.5;0.75,best_0.75 | Likelihood of the coin bias p after H, H, T, H, scaled so its peak is 1. It is highest at p = 0.75.
~~~

```
p = np.array([0.5, 0.75])
print(np.round(p ** 3 * (1 - p), 4), round(1000 * np.log(0.1), 1))  # [0.0625 0.1055] -2302.6
```

A product of 1000 probabilities of 0.1 underflows to 0 in float64; its log is just <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>2302.6</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

the likelihood scores parameter values by how probable they make the observed data.

</div>

## Where this breaks in practice

- Never multiply raw probabilities over a dataset; sum log-probabilities.
