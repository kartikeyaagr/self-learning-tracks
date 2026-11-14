---
id: "09.41"
course: "ml-maths-course"
module: "09"
slug: "the-definite-integral"
title: "The Definite Integral"
keyIdeas: ["The definite integral is the limit of Riemann sums: ⟦2971⟧.", "It is a signed area: parts below the axis count negative.", "It is linear: ⟦2972⟧, and additive over adjacent intervals."]
question: "Compute ⟦2969⟧ and ⟦2970⟧ from the picture."
answer: "Step 1, ⟦7⟧ is negative on ⟦2977⟧ and positive on ⟦4⟧ with equal triangles of area ⟦2547⟧: they cancel, so ⟦2978⟧. Step 2, ⟦2561⟧ gives two positive triangles: ⟦2979⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["From partitions to volume"]
tools: ["NumPy"]
topic: {"id": "09.06", "title": "From partitions to volume"}
---

## The big picture

The limit of Riemann sums deserves a name and notation. Its properties (linearity, additivity, sign) are what make integrals easy to manipulate.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msubsup><mo>∫</mo><mi>a</mi><mi>b</mi></msubsup><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><munder><mrow><mi>lim</mi><mo>⁡</mo></mrow><mrow><mi>n</mi><mo>→</mo><mi mathvariant="normal">∞</mi></mrow></munder><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>f</mi><mo stretchy="false">(</mo><msubsup><mi>x</mi><mi>i</mi><mo>∗</mo></msubsup><mo stretchy="false">)</mo><mtext> </mtext><mi mathvariant="normal">Δ</mi><mi>x</mi></mrow></math></span></span>
For continuous <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> the limit exists and does not depend on how the points <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>x</mi><mi>i</mi><mo>∗</mo></msubsup></mrow></math></span> are chosen.

~~~dg-raw
%%viz f_curve f=x g=Math.abs(x) x=-1,1 y=-1,1 tk=0.5 fl=x:_areas_cancel gl=∣x∣:_areas_add | ∫x dx over [−1, 1] is 0, because the area below the axis counts as negative and cancels the area above. ∫|x| dx is 1.
~~~

## Properties

| Property | Formula |
| --- | --- |
| linear | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∫</mo><mo stretchy="false">(</mo><mi>α</mi><mi>f</mi><mo>+</mo><mi>β</mi><mi>g</mi><mo stretchy="false">)</mo><mo>=</mo><mi>α</mi><mo>∫</mo><mi>f</mi><mo>+</mo><mi>β</mi><mo>∫</mo><mi>g</mi></mrow></math></span> |
| additive | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mi>a</mi><mi>b</mi></msubsup><mi>f</mi><mo>+</mo><msubsup><mo>∫</mo><mi>b</mi><mi>c</mi></msubsup><mi>f</mi><mo>=</mo><msubsup><mo>∫</mo><mi>a</mi><mi>c</mi></msubsup><mi>f</mi></mrow></math></span> |
| monotone | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>≤</mo><mi>g</mi><mo>⇒</mo><mo>∫</mo><mi>f</mi><mo>≤</mo><mo>∫</mo><mi>g</mi></mrow></math></span> |
| reversed limits | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mi>b</mi><mi>a</mi></msubsup><mi>f</mi><mo>=</mo><mo>−</mo><msubsup><mo>∫</mo><mi>a</mi><mi>b</mi></msubsup><mi>f</mi></mrow></math></span> |

```
x = np.linspace(-1, 1, 200001); dx = x[1] - x[0]
print(round((x * dx).sum(), 6) + 0, round((np.abs(x) * dx).sum(), 4))  # 0.0 1.0
```

<div class="take">
<div class="lab">Key takeaway</div>

the definite integral is the limit of Riemann sums: a signed area that is linear and additive.

</div>
