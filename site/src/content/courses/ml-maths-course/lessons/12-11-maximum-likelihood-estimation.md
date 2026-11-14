---
id: "12.11"
course: "ml-maths-course"
module: "12"
slug: "maximum-likelihood-estimation"
title: "Maximum Likelihood Estimation"
keyIdeas: ["⟦3897⟧: pick the parameters that make the data most probable.", "For simple models, set the derivative to zero; for neural networks, run gradient descent on ⟦3898⟧.", "The Bernoulli MLE is ⟦3899⟧; the Poisson MLE is ⟦3900⟧."]
question: "Derive the MLE of ⟦102⟧ from ⟦1199⟧ successes in ⟦233⟧ trials."
answer: "Step 1, ⟦3906⟧. Step 2, ⟦3907⟧ gives ⟦3899⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Likelihood-based estimation"]
tools: ["SciPy"]
topic: {"id": "12.02", "title": "Likelihood-based estimation"}
---

## The big picture

Maximum likelihood is the default way to fit a probabilistic model, including every network trained with cross-entropy.

## The recipe

1. Write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℓ</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mo>∑</mo><mi>i</mi></msub><mi>log</mi><mo>⁡</mo><mi>p</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>∣</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></math></span>.
2. Maximise it: solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi mathvariant="normal">ℓ</mi><mo>=</mo><mn>0</mn></mrow></math></span>, or run gradient ascent.

```
from scipy.optimize import minimize_scalar
x = np.array([2, 0, 3, 1, 4])
nll = lambda lam: -(x * np.log(lam) - lam).sum()
print(round(minimize_scalar(nll, bounds=(0.01, 10), method='bounded').x, 3))  # 2.0
```

Minimising the Poisson negative log-likelihood lands on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover><mo>=</mo><mn>2</mn></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=-(10*Math.log(x)-5*x) x=0.5,5 y=2,12 tk=1 pts=2,minimum_at_x̄_=_2 | Poisson negative log-likelihood for counts 2, 0, 3, 1, 4 (constant terms dropped). Its minimum is at the sample mean, 2.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**zero successes**

With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>=</mo><mn>0</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>=</mo><mn>0</mn></mrow></math></span>, and the model says an event that can happen is impossible. A prior fixes this (lesson 12.15).

</div>

<div class="take">
<div class="lab">Key takeaway</div>

MLE maximises the log-likelihood; for simple models the answer is a familiar average.

</div>

## Where this breaks in practice

- MLE overfits when parameters are many and data are few.
