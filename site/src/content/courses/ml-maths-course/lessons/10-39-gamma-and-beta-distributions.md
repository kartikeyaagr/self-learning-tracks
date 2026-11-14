---
id: "10.39"
course: "ml-maths-course"
module: "10"
slug: "gamma-and-beta-distributions"
title: "Gamma and Beta Distributions"
keyIdeas: ["Gamma(⟦3579⟧) lives on ⟦279⟧ with mean ⟦3580⟧ and variance ⟦3581⟧; it models positive scales and rates.", "Beta(⟦3582⟧) lives on ⟦4⟧ with mean ⟦3583⟧; it models uncertainty about a probability.", "Match the distribution to the kind of value: positive, between 0 and 1, or any real."]
question: "What is the mean of Beta(2, 8)? Of Gamma with ⟦3577⟧, ⟦3578⟧?"
answer: "Step 1, ⟦3586⟧. Step 2, ⟦230⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.06", "title": "Continuous probability distributions"}
---

## The big picture

A probability must stay in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span> and a rate must be positive. Gaussians ignore those limits; Beta and Gamma respect them.

## Shapes

| Distribution | Values | Mean | Variance |
| --- | --- | --- | --- |
| Gamma(<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo separator="true">,</mo><mi>θ</mi></mrow></math></span>) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mi>θ</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><msup><mi>θ</mi><mn>2</mn></msup></mrow></math></span> |
| Beta(<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>α</mi><mo separator="true">,</mo><mi>β</mi></mrow></math></span>) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mi>α</mi><mrow><mi>α</mi><mo>+</mo><mi>β</mi></mrow></mfrac></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mrow><mi>α</mi><mi>β</mi></mrow><mrow><mo stretchy="false">(</mo><mi>α</mi><mo>+</mo><mi>β</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo stretchy="false">(</mo><mi>α</mi><mo>+</mo><mi>β</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mfrac></mrow></math></span> |

~~~dg-raw
%%viz f_curve f=72*x*Math.pow(1-x,7) x=0,1 y=0,3.5 tk=0.25 | Beta(2, 8) stays inside [0, 1], as a probability must. It peaks at 0.125 and its mean is 0.2.
~~~

```
from scipy import stats
print(stats.beta(2, 8).mean(), stats.gamma(2, scale=3).mean(), stats.gamma(2, scale=3).var())  # 0.2 6.0 18.0
```

<div class="take">
<div class="lab">Key takeaway</div>

Beta for probabilities, Gamma for positive quantities.

</div>

## Where this breaks in practice

- SciPy and NumPy use shape and scale for Gamma; many papers and `torch.distributions.Gamma` use shape and rate.
