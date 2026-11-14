---
id: "09.23"
course: "ml-maths-course"
module: "09"
slug: "gradients-are-perpendicular-to-level-sets"
title: "Gradients Are Perpendicular to Level Sets"
keyIdeas: ["Along a level set ⟦90⟧ does not change, so the directional derivative along it is 0.", "Therefore ⟦2698⟧ is perpendicular to the level set through the point (when ⟦2725⟧).", "This explains why gradient descent zig-zags across long, thin contours."]
question: "For ⟦2724⟧ at ⟦849⟧ on the level set ⟦2477⟧, find ⟦2698⟧ and the tangent direction of the curve."
answer: "Step 1, ⟦2728⟧. Step 2, at ⟦849⟧ the ellipse is at its rightmost point, so its tangent is vertical, ⟦5⟧. Step 3, ⟦2729⟧: perpendicular."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Gradients and vector-valued maps"]
tools: ["NumPy"]
topic: {"id": "09.03", "title": "Gradients and vector-valued maps"}
---

## The big picture

A contour map and the gradient are tightly linked: the gradient always crosses the contour lines at right angles. Seeing this explains the zig-zag paths of gradient descent on stretched bowls.

## Why

If you walk along a level set, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> stays constant, so the rate of change in that direction is zero: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo>⋅</mo><mi>u</mi><mo>=</mo><mn>0</mn></mrow></math></span> for every tangent direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span>. A vector with zero dot product against every tangent direction is perpendicular to the curve.

~~~dg-raw
%%viz f_plane x=-3,3 y=-1.5,2.3 tk=1 pg=2.00:0.00_1.98:0.16_1.90:0.31_1.78:0.45_1.62:0.59_1.41:0.71_1.18:0.81_0.91:0.89_0.62:0.95_0.31:0.99_0.00:1.00_-0.31:0.99_-0.62:0.95_-0.91:0.89_-1.18:0.81_-1.41:0.71_-1.62:0.59_-1.78:0.45_-1.90:0.31_-1.98:0.16_-2.00:0.00_-1.98:-0.16_-1.90:-0.31_-1.78:-0.45_-1.62:-0.59_-1.41:-0.71_-1.18:-0.81_-0.91:-0.89_-0.62:-0.95_-0.31:-0.99_0.00:-1.00_0.31:-0.99_0.62:-0.95_0.91:-0.89_1.18:-0.81_1.41:-0.71_1.62:-0.59_1.78:-0.45_1.90:-0.31_1.98:-0.16,a pt=1.41,0.71,,k va=1.41,0.71,1.99,1.87,∇f,f | The level set x² + 4y² = 4 and the gradient at (√2, 1/√2). The gradient (2√2, 4√2) points straight out of the ellipse, at right angles to it.
~~~

```
p = np.array([np.sqrt(2), 1 / np.sqrt(2)])
grad = np.array([2 * p[0], 8 * p[1]])
tangent = np.array([-2 * p[1], p[0] / 2])
print(round(grad @ tangent, 10) + 0)  # 0.0
```

The tangent of the ellipse <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mi>cos</mi><mo>⁡</mo><mi>t</mi><mo separator="true">,</mo><mi>sin</mi><mo>⁡</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>2</mn><mi>sin</mi><mo>⁡</mo><mi>t</mi><mo separator="true">,</mo><mi>cos</mi><mo>⁡</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></math></span>; its dot product with the gradient is 0.

## Zig-zags

On a long thin ellipse, the gradient points mostly across the valley, not along it toward the minimum. Each step overshoots across, so the path zig-zags (lesson 07.06).

<div class="take">
<div class="lab">Key takeaway</div>

the gradient crosses level sets at right angles, pointing toward higher values.

</div>
