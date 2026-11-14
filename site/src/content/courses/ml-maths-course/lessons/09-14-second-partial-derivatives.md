---
id: "09.14"
course: "ml-maths-course"
module: "09"
slug: "second-partial-derivatives"
title: "Second Partial Derivatives"
keyIdeas: ["Differentiating twice gives ⟦2600⟧, ⟦2601⟧ and the mixed partials ⟦2602⟧, ⟦2603⟧.", "If the second partials are continuous, the mixed ones are equal (Schwarz's theorem), so the Hessian is symmetric.", "Second partials measure curvature: how fast the slope itself changes."]
question: "For ⟦2566⟧, compute all four second partials."
answer: "Step 1, ⟦2579⟧, ⟦2582⟧. Step 2, ⟦2609⟧, ⟦2610⟧. Step 3, ⟦2611⟧ and ⟦2612⟧. Equal, as Schwarz's theorem says."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

First partials give slopes. Second partials say how those slopes change, which is curvature. Collected into the Hessian matrix, they decide whether a critical point is a minimum (module 07).

## Four second partials in 2D

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>f</mi><mrow><mi>x</mi><mi>x</mi></mrow></msub><mo>=</mo><mfrac><mrow><msup><mi mathvariant="normal">∂</mi><mn>2</mn></msup><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msup><mi>x</mi><mn>2</mn></msup></mrow></mfrac><mo separator="true">,</mo><mspace width="1em"/><msub><mi>f</mi><mrow><mi>y</mi><mi>y</mi></mrow></msub><mo>=</mo><mfrac><mrow><msup><mi mathvariant="normal">∂</mi><mn>2</mn></msup><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msup><mi>y</mi><mn>2</mn></msup></mrow></mfrac><mo separator="true">,</mo><mspace width="1em"/><msub><mi>f</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub><mo>=</mo><mfrac><mi mathvariant="normal">∂</mi><mrow><mi mathvariant="normal">∂</mi><mi>y</mi></mrow></mfrac><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><mi>x</mi></mrow></mfrac><mo separator="true">,</mo><mspace width="1em"/><msub><mi>f</mi><mrow><mi>y</mi><mi>x</mi></mrow></msub><mo>=</mo><mfrac><mi mathvariant="normal">∂</mi><mrow><mi mathvariant="normal">∂</mi><mi>x</mi></mrow></mfrac><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><mi>y</mi></mrow></mfrac></mrow></math></span></span>

## Symmetry

**Schwarz's theorem**: if the second partials are continuous, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub><mo>=</mo><msub><mi>f</mi><mrow><mi>y</mi><mi>x</mi></mrow></msub></mrow></math></span>. So the **Hessian**
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>H</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>f</mi><mrow><mi>x</mi><mi>x</mi></mrow></msub></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>f</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>f</mi><mrow><mi>y</mi><mi>x</mi></mrow></msub></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>f</mi><mrow><mi>y</mi><mi>y</mi></mrow></msub></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span></span>
is symmetric, and the spectral theorem (lesson 05.10) applies.

```
f = lambda x, y: x ** 2 * y + np.sin(y)
x, y, h = 1.0, 0.5, 1e-4
fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h)
print(round(fxy, 4), 2 * x)  # 2.0 2.0
```

## Reading curvature

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mrow><mi>x</mi><mi>x</mi></mrow></msub><mo>&gt;</mo><mn>0</mn></mrow></math></span>: the slice in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> curves up.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span>: the slope in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> changes as you move in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>. The variables interact.

<div class="take">
<div class="lab">Key takeaway</div>

second partials measure curvature; the mixed ones agree, so the Hessian is symmetric.

</div>
