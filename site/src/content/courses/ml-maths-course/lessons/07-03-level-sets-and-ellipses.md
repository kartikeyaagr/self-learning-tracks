---
id: "07.03"
course: "ml-maths-course"
module: "07"
slug: "level-sets-and-ellipses"
title: "Level Sets and Ellipses"
keyIdeas: ["For positive definite ⟦8⟧, the level set ⟦2156⟧ is an ellipse (ellipsoid in higher dimensions).", "Its axes point along the eigenvectors; the half-length along eigenvector ⟦274⟧ is ⟦2157⟧.", "Large eigenvalue: short axis and steep walls. Small eigenvalue: long axis and a flat valley."]
question: "For ⟦2155⟧, where does the level set ⟦2156⟧ cross each axis? How elongated is it?"
answer: "Step 1, ⟦2164⟧. Step 2, on the ⟦1370⟧-axis, ⟦2165⟧ gives ⟦2166⟧. Step 3, on the ⟦697⟧-axis, ⟦2167⟧ gives ⟦2168⟧. Step 4, the ellipse is 10 times longer than it is tall; the ratio is ⟦2169⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Curvature and quadratic objectives"]
tools: ["NumPy"]
topic: {"id": "07.01", "title": "Curvature and quadratic objectives"}
---

## The big picture

A contour map of a loss shows its level sets. Near a minimum they are ellipses, and their shape predicts how gradient descent will behave: round contours are easy, long thin ones are slow.

## Axes from eigenvalues

In eigen-coordinates the level set is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>∑</mo><mi>i</mi></msub><msub><mi>λ</mi><mi>i</mi></msub><msubsup><mi>y</mi><mi>i</mi><mn>2</mn></msubsup><mo>=</mo><mn>1</mn></mrow></math></span>. Setting all but one <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>y</mi><mi>i</mi></msub></mrow></math></span> to zero gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>y</mi><mi>i</mi></msub><mo>=</mo><mo>±</mo><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><msub><mi>λ</mi><mi>i</mi></msub></msqrt></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-2,2 y=-2,2 tk=1 pg=0.82:0.82_1.05:0.56_1.25:0.28_1.41:0.00_1.53:-0.28_1.61:-0.56_1.63:-0.82_1.61:-1.05_1.53:-1.25_1.41:-1.41_1.25:-1.53_1.05:-1.61_0.82:-1.63_0.56:-1.61_0.28:-1.53_0.00:-1.41_-0.28:-1.25_-0.56:-1.05_-0.82:-0.82_-1.05:-0.56_-1.25:-0.28_-1.41:0.00_-1.53:0.28_-1.61:0.56_-1.63:0.82_-1.61:1.05_-1.53:1.25_-1.41:1.41_-1.25:1.53_-1.05:1.61_-0.82:1.63_-0.56:1.61_-0.28:1.53_0.00:1.41_0.28:1.25_0.56:1.05,m;0.41:0.41_0.52:0.28_0.63:0.14_0.71:0.00_0.77:-0.14_0.80:-0.28_0.82:-0.41_0.80:-0.52_0.77:-0.63_0.71:-0.71_0.63:-0.77_0.52:-0.80_0.41:-0.82_0.28:-0.80_0.14:-0.77_0.00:-0.71_-0.14:-0.63_-0.28:-0.52_-0.41:-0.41_-0.52:-0.28_-0.63:-0.14_-0.71:0.00_-0.77:0.14_-0.80:0.28_-0.82:0.41_-0.80:0.52_-0.77:0.63_-0.71:0.71_-0.63:0.77_-0.52:0.80_-0.41:0.82_-0.28:0.80_-0.14:0.77_0.00:0.71_0.14:0.63_0.28:0.52,a v=0.41,0.41,λ=3,f;0.71,-0.71,λ=1,b | Level sets q = 1 (green) and q = 4 (grey) of A = [[2, 1], [1, 2]]. The short axis is along the eigenvector (1, 1) with λ = 3 (half-length 1/√3 ≈ 0.58); the long axis is along (1, −1) with λ = 1 (half-length 1).
~~~

Doubling the level from 1 to 4 doubles every axis, since <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi><mo stretchy="false">(</mo><mn>2</mn><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>4</mn><mi>q</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>.

```
A = np.array([[2.0, 1], [1, 2]])
lam, Q = np.linalg.eigh(A)
print(np.round(1 / np.sqrt(lam), 4))  # [1.     0.5774]
p = Q[:, 0] / np.sqrt(lam[0])
print(round(p @ A @ p, 6))  # 1.0
```

## Elongation and the condition number

The ratio of longest to shortest axis is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mi>min</mi><mo>⁡</mo></msub></mrow></msqrt><mo>=</mo><msqrt><mi>κ</mi></msqrt></mrow></math></span>. A loss with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi><mo>=</mo><mn>1</mn><msup><mn>0</mn><mn>4</mn></msup></mrow></math></span> has contours 100 times longer than wide. Gradients point mostly across the valley, not along it, which is why plain gradient descent zig-zags (lesson 07.06).

<div class="take">
<div class="lab">Key takeaway</div>

level sets of a positive definite quadratic are ellipses with axes along the eigenvectors and half-lengths <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><msub><mi>λ</mi><mi>i</mi></msub></msqrt></mrow></math></span>.

</div>
