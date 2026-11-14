---
id: "07.02"
course: "ml-maths-course"
module: "07"
slug: "reading-shape-from-eigenvalues"
title: "Reading Shape from Eigenvalues"
keyIdeas: ["In the eigenvector basis ⟦2143⟧, a quadratic form has no cross terms: ⟦2144⟧.", "All ⟦1762⟧: a bowl. All ⟦2145⟧: an upside-down bowl. Mixed signs: a saddle. A zero: a flat valley.", "The size of ⟦1622⟧ is the steepness along eigenvector ⟦274⟧."]
question: "Classify ⟦1765⟧ for ⟦2142⟧ and give the directions where it curves up and down."
answer: "Step 1, trace 2, determinant ⟦2152⟧, so ⟦2153⟧. Step 2, ⟦1586⟧ with eigenvector ⟦111⟧ and ⟦2154⟧ with ⟦1352⟧. Step 3, mixed signs: a saddle, curving up along ⟦111⟧ and down along ⟦1352⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Curvature and quadratic objectives"]
tools: ["NumPy"]
topic: {"id": "07.01", "title": "Curvature and quadratic objectives"}
---

## The big picture

The cross term <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><msub><mi>x</mi><mn>2</mn></msub></mrow></math></span> hides the shape. Rotating to the eigenvectors (module 05) removes it, and the eigenvalues then tell you everything.

## Diagonalising the form

Write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>Q</mi><mi mathvariant="normal">Λ</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> and set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span>. Then
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi><mi mathvariant="normal">Λ</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><msup><mi>y</mi><mi mathvariant="normal">⊤</mi></msup><mi mathvariant="normal">Λ</mi><mi>y</mi><mo>=</mo><munder><mo>∑</mo><mi>i</mi></munder><msub><mi>λ</mi><mi>i</mi></msub><msubsup><mi>y</mi><mi>i</mi><mn>2</mn></msubsup></mrow></math></span></span>
Each eigen-direction contributes on its own.

| Eigenvalues | Shape | Name |
| --- | --- | --- |
| all positive | bowl, minimum at 0 | positive definite |
| all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>≥</mo><mn>0</mn></mrow></math></span>, some zero | valley, flat along the zero directions | positive semidefinite |
| mixed signs | saddle | indefinite |
| all negative | dome, maximum at 0 | negative definite |

## Slices through a saddle

~~~dg-raw
%%viz f_curve f=3*x*x g=-x*x x=-1.5,1.5 y=-2.5,7 tk=1 fl=along_(1~1):_3t² gl=along_(1~−1):_−t² | Slices of q = x₁² + 4x₁x₂ + x₂² through the origin, along unit eigenvectors. Along (1, 1) it curves up with λ = 3; along (1, −1) it curves down with λ = −1.
~~~

Along a unit eigenvector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>t</mi><mi>q</mi></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>=</mo><mi>λ</mi><msup><mi>t</mi><mn>2</mn></msup></mrow></math></span>: a parabola whose curvature is the eigenvalue.

```
A = np.array([[1.0, 2], [2, 1]])
lam, Q = np.linalg.eigh(A)
print(lam)  # [-1.  3.]
for i in range(2):
    v = Q[:, i]; print(round(v @ A @ v, 6))
```

<div class="take">
<div class="lab">Key takeaway</div>

in the eigenvector basis a quadratic form is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∑</mo><msub><mi>λ</mi><mi>i</mi></msub><msubsup><mi>y</mi><mi>i</mi><mn>2</mn></msubsup></mrow></math></span>; the signs of the eigenvalues give bowl, saddle or dome, and their sizes give the steepness.

</div>
