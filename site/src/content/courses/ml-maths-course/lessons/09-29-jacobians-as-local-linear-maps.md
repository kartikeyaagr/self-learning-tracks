---
id: "09.29"
course: "ml-maths-course"
module: "09"
slug: "jacobians-as-local-linear-maps"
title: "Jacobians as Local Linear Maps"
keyIdeas: ["Near ⟦7⟧, ⟦2800⟧: every differentiable map looks linear up close.", "So the matrix tools of modules 03 to 07 apply locally: rank, singular values and determinant of ⟦1726⟧ describe how ⟦2801⟧ stretches space near ⟦7⟧.", "The approximation error shrinks like ⟦2802⟧."]
question: "For ⟦2784⟧ at ⟦1049⟧, predict ⟦2799⟧ with the Jacobian and compare with the exact value."
answer: "Step 1, ⟦2805⟧, ⟦2798⟧. Step 2, ⟦2806⟧. Step 3, prediction ⟦2807⟧; exact ⟦2808⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Jacobians and local approximation"]
tools: ["NumPy"]
topic: {"id": "09.04", "title": "Jacobians and local approximation"}
---

## The big picture

A curved map bends a grid into curved lines. Zoom in on one small square and the image is almost a parallelogram: the image of the square under the matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>J</mi></mrow></math></span>. That is what "differentiable" means for maps.

## Linearisation

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>d</mi><mo stretchy="false">)</mo><mo>≈</mo><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>+</mo><msub><mi>J</mi><mi>F</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi></mrow></math></span></span>

~~~dg-raw
%%viz jac | Near p, a curved map acts like a matrix. Shrink the square and the dashed parallelogram fits better.
~~~

```
F = lambda v: np.array([v[0] ** 2 * v[1], v[0] + 3 * v[1]])
J = np.array([[4.0, 1], [1, 3]]); x = np.array([1.0, 2])
for s in (0.1, 0.01):
    d = np.array([s, s])
    print(s, f"{np.linalg.norm(F(x + d) - F(x) - J @ d):.1e}")
```

Ten times smaller step, a hundred times smaller error.

## What <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>J</mi></mrow></math></span> tells you locally

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi>J</mi><mi mathvariant="normal">∣</mi></mrow></math></span>: how much small areas are scaled (here <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>12</mn><mo>−</mo><mn>1</mn><mo>=</mo><mn>11</mn></mrow></math></span>).
- Singular values: the most and least stretch of small displacements.
- Rank deficiency: some directions are crushed, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>F</mi></mrow></math></span> is not locally invertible.

<div class="take">
<div class="lab">Key takeaway</div>

near any point a differentiable map acts like its Jacobian matrix.

</div>
