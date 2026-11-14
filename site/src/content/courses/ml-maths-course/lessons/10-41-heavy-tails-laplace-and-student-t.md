---
id: "10.41"
course: "ml-maths-course"
module: "10"
slug: "heavy-tails-laplace-and-student-t"
title: "Heavy Tails: Laplace and Student-t"
keyIdeas: ["The tail is how fast the density falls far from the centre; it decides how a model treats outliers.", "Gaussian loss grows like ⟦3596⟧, Laplace like ⟦3597⟧, Student-⟦166⟧ only like ⟦3598⟧.", "That is the probabilistic reason squared error is sensitive to outliers and absolute error is robust."]
question: "A residual is 10 standard deviations. How much extra negative log-density does it cost under a Gaussian, a Laplace and a Student-⟦3595⟧?"
answer: "Step 1, Gaussian: ⟦3601⟧. Step 2, Laplace: 10. Step 3, ⟦3595⟧: about 7."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.06", "title": "Continuous probability distributions"}
---

## The big picture

Picking the error distribution is the same as picking the loss. Heavy-tailed distributions expect occasional large errors and don't let them drag the fit.

## Tail weight

~~~dg-flow Tails from light to heavy.
Uniform (no tails) -(heavier)-> Gaussian tails e^(-x²/2) -(heavier)-> Laplace tails e^(-|x|) -(heavier)-> *Student-t tails, polynomial
~~~

~~~dg-raw
%%viz f_curve f=x*x/2 g=Math.abs(x) x=-6,6 y=0,12 tk=2 fl=Gaussian:_r²/2 gl=Laplace:_∣r∣ | The loss each point pays under two noise models. A point at r = 6 costs 18 under the Gaussian and only 6 under the Laplace.
~~~

```
from scipy import stats
for name, d in [('normal', stats.norm()), ('laplace', stats.laplace()), ('t3', stats.t(3))]:
    print(name, round(d.logpdf(0) - d.logpdf(10), 1))
# normal 50.0
# laplace 10.0
# t3 7.1
```

<div class="take">
<div class="lab">Key takeaway</div>

tail weight is a modelling choice about outliers: Gaussian punishes them, Laplace and Student-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> tolerate them.

</div>

## Where this breaks in practice

- Student-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ν</mi><mo>≤</mo><mn>2</mn></mrow></math></span> has infinite variance, and with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ν</mi><mo>≤</mo><mn>1</mn></mrow></math></span> no mean.
