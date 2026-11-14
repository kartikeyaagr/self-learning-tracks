---
id: "09.17"
course: "ml-maths-course"
module: "09"
slug: "linear-approximation-and-error"
title: "Linear Approximation and Error"
keyIdeas: ["⟦2638⟧ predicts how much a small step changes ⟦90⟧.", "The error of this first-order prediction is about ⟦2137⟧, so it grows with the square of the step.", "Gradient descent trusts the linear prediction; the step size must keep the quadratic error small."]
question: "⟦2451⟧ at ⟦1049⟧ with step ⟦2637⟧. Predict the change in ⟦90⟧ linearly, then compare with the true change."
answer: "Step 1, ⟦2644⟧, so the predicted change is ⟦2645⟧. Step 2, the true value at ⟦2646⟧ is ⟦2647⟧, a change of ⟦2648⟧. Step 3, the gap ⟦2649⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

The gradient lets you predict the effect of a small change without recomputing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span>. That prediction is what gradient descent acts on, and its error explains why steps must be small.

## The first-order prediction

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">Δ</mi><mi>f</mi><mo>≈</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mi>d</mi></mrow></math></span></span>
Its error is the second-order term from lesson 07.04:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>d</mi><mo stretchy="false">)</mo><mo>−</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>−</mo><mi mathvariant="normal">∇</mi><msup><mi>f</mi><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mo>≈</mo><mstyle displaystyle="false" scriptlevel="0"><mfrac><mn>1</mn><mn>2</mn></mfrac></mstyle><msup><mi>d</mi><mi mathvariant="normal">⊤</mi></msup><mi>H</mi><mi>d</mi></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=x*x g=2*x-1 x=-0.5,2.5 y=-1,6 tk=1 pts=1,x_=_1 fl=f(x)_=_x² gl=tangent_line_2x_−_1 | In one variable: near x = 1 the tangent line is a good prediction; further away the gap grows like the square of the distance.
~~~

```
f = lambda v: (v ** 2).sum()
x = np.array([1.0, 2]); g = 2 * x
for s in (1.0, 0.1, 0.01):
    d = s * np.array([0.1, -0.2])
    print(s, round(g @ d, 6), round(f(x + d) - f(x), 6))
```

For small steps the predicted and true changes agree almost exactly; for larger ones the quadratic term shows.

## Why steps must be small

Gradient descent picks <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>=</mo><mo>−</mo><mi>η</mi><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span>, which gives a predicted drop of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span>. The curvature term adds back about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mi>η</mi><mn>2</mn></msup><mi mathvariant="normal">∇</mi><msup><mi>f</mi><mi mathvariant="normal">⊤</mi></msup><mi>H</mi><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span>. If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi></mrow></math></span> is too large, the second term wins and the loss goes up: the learning-rate limit of lesson 07.06.

<div class="take">
<div class="lab">Key takeaway</div>

the gradient predicts small changes to first order; the error grows with the square of the step, which caps the learning rate.

</div>
