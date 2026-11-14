---
id: "09.18"
course: "ml-maths-course"
module: "09"
slug: "the-multivariable-chain-rule"
title: "The Multivariable Chain Rule"
keyIdeas: ["If ⟦2449⟧ with ⟦2654⟧ and ⟦2655⟧, then ⟦2656⟧.", "Each path from ⟦166⟧ to ⟦313⟧ contributes the product of the derivatives along it; contributions add.", "In vector form, ⟦2657⟧."]
question: "⟦2650⟧ with ⟦2651⟧, ⟦2652⟧. Find ⟦2653⟧ at ⟦778⟧ by the chain rule and check directly."
answer: "Step 1, at ⟦778⟧: ⟦335⟧, ⟦2445⟧. Step 2, path through ⟦7⟧: ⟦2663⟧. Path through ⟦509⟧: ⟦2664⟧. Step 3, total 9. Step 4, directly ⟦2665⟧, so ⟦2666⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

In a network every quantity depends on earlier ones through many routes. The chain rule says how to combine the local derivatives: multiply along each route, add over routes. Backpropagation is this rule, organised efficiently.

## The rule

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mrow><mi>d</mi><mi>z</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><mi>x</mi></mrow></mfrac><mfrac><mrow><mi>d</mi><mi>x</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>+</mo><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><mi>y</mi></mrow></mfrac><mfrac><mrow><mi>d</mi><mi>y</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac></mrow></math></span></span>

~~~dg-flow Chain rule along a path: each edge multiplies by a local derivative, paths add.
t -(dx/dt)-> x -(df/dx)-> *z = f(x, y)
t -(dy/dt)-> y -(df/dy)-> *z = f(x, y)
~~~

```
z = lambda t: (t ** 2) * (3 * t)
h = 1e-6
print(round((z(1 + h) - z(1 - h)) / (2 * h), 4))  # 9.0
```

## Moving along a curve

Walking along a path <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></math></span> through a landscape, the height changes at rate
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mi>d</mi><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">∇</mi><msup><mi>f</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></math></span></span>
the directional derivative scaled by your speed. Along gradient descent's continuous version, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>=</mo><mo>−</mo><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span>, the loss falls at rate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

the chain rule multiplies local derivatives along each path and adds over paths.

</div>
