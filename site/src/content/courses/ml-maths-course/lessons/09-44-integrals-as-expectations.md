---
id: "09.44"
course: "ml-maths-course"
module: "09"
slug: "integrals-as-expectations"
title: "Integrals as Expectations"
keyIdeas: ["The expectation of ⟦3014⟧ for a density ⟦102⟧ is ⟦3015⟧: a weighted average.", "A density is normalised when ⟦3016⟧; the Gaussian needs ⟦3017⟧.", "Expected loss, marginal likelihood, normalising constants and KL divergence are all integrals."]
question: "For the density ⟦3011⟧ on ⟦4⟧, check normalisation and find ⟦3012⟧ and ⟦3013⟧."
answer: "Step 1, ⟦3025⟧. Step 2, ⟦3026⟧. Step 3, ⟦3027⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["From partitions to volume"]
tools: ["NumPy", "SciPy"]
topic: {"id": "09.06", "title": "From partitions to volume"}
---

## The big picture

In ML you rarely want an area for its own sake. You want **averages**: expected loss over all data, expected reward, the probability of an event. Each one is an integral of the form <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∫</mo><mi>g</mi><mtext> </mtext><mi>p</mi></mrow></math></span>.

## Weighted averages

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>g</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo><mo>=</mo><mo>∫</mo><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span></span>
Each value <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> is weighted by how likely <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is.

```
from scipy.integrate import quad
p = lambda x: 2 * x
print([round(quad(lambda x: x ** k * p(x), 0, 1)[0], 6) for k in (0, 1, 2)])  # [1.0, 0.666667, 0.5]
print(np.isclose(quad(lambda x: np.exp(-x ** 2 / 2), -np.inf, np.inf)[0], np.sqrt(2 * np.pi)))  # True
```

## Integrals everywhere in ML

| Quantity | Integral |
| --- | --- |
| expected loss (risk) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∫</mo><mi mathvariant="normal">ℓ</mi><mo stretchy="false">(</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi></mrow></math></span> |
| marginal likelihood | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∫</mo><mi>p</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>θ</mi><mo stretchy="false">)</mo><mi>p</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>θ</mi></mrow></math></span> |
| normalising constant | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Z</mi><mo>=</mo><mo>∫</mo><mover accent="true"><mi>p</mi><mo>~</mo></mover><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span> |
| KL divergence | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∫</mo><mi>p</mi><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>p</mi><mi mathvariant="normal">/</mi><mi>q</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span> |

Most cannot be computed exactly. Training loss is a sample average standing in for the risk; variational methods bound integrals; MCMC samples them.

<div class="take">
<div class="lab">Key takeaway</div>

in ML an integral is almost always an expectation: a probability-weighted average.

</div>

## Where this breaks in practice

- Some integrals are infinite. The Cauchy distribution has no mean, so sample averages never settle down.
