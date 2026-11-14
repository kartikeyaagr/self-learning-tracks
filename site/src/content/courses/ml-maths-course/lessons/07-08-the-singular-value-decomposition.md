---
id: "07.08"
course: "ml-maths-course"
module: "07"
slug: "the-singular-value-decomposition"
title: "The Singular Value Decomposition"
keyIdeas: ["Every ⟦961⟧ matrix factors as ⟦2252⟧ with orthogonal ⟦207⟧, ⟦1204⟧ and non-negative diagonal ⟦930⟧.", "Read right to left: rotate by ⟦2253⟧, stretch the axes by ⟦2244⟧, rotate by ⟦207⟧. Direction by direction, ⟦2254⟧.", "⟦1204⟧ holds eigenvectors of ⟦1997⟧, ⟦207⟧ those of ⟦2255⟧, and ⟦2256⟧ are their eigenvalues."]
question: "Find the singular values of ⟦2233⟧ by hand via ⟦1997⟧, and check two identities."
answer: "Step 1, ⟦2262⟧. Step 2, trace 50, determinant 225, so ⟦2263⟧ and ⟦2264⟧. Step 3, ⟦2265⟧, ⟦2266⟧. Step 4, checks: ⟦2267⟧ and ⟦2268⟧, the sum of squared entries."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Singular value decomposition"]
tools: ["NumPy"]
topic: {"id": "07.02", "title": "Singular value decomposition"}
---

## The big picture

The previous lesson said what singular values are. The SVD says that this rotate-stretch-rotate picture describes every matrix exactly, and gives you the rotations too.

## The factorisation

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>=</mo><mi>U</mi><mi mathvariant="normal">Σ</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span></span>

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>n</mi></mrow></math></span>, orthogonal): columns <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mi>i</mi></msub></mrow></math></span> are the input directions.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span> (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>n</mi></mrow></math></span>): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>1</mn></msub><mo>≥</mo><msub><mi>σ</mi><mn>2</mn></msub><mo>≥</mo><mo>⋯</mo><mo>≥</mo><mn>0</mn></mrow></math></span> on the diagonal, zeros elsewhere.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>m</mi></mrow></math></span>, orthogonal): columns <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mi>i</mi></msub></mrow></math></span> are the output directions.
The defining property is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><msub><mi>v</mi><mi>i</mi></msub><mo>=</mo><msub><mi>σ</mi><mi>i</mi></msub><msub><mi>u</mi><mi>i</mi></msub></mrow></math></span>: input direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mi>i</mi></msub></mrow></math></span> goes to output direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mi>i</mi></msub></mrow></math></span>, stretched by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span>.

~~~dg-raw
%%viz m_svd live | Any 2 × 2 matrix as rotate, stretch, rotate. Drag Stage or press Play to see the three steps one at a time.
~~~

```
A = np.array([[3.0, 0], [4, 5]])
U, s, Vt = np.linalg.svd(A)
print(np.round(s, 4))  # [6.7082 2.2361]
print(np.allclose(U @ np.diag(s) @ Vt, A))  # True
print(np.allclose(A @ Vt[0], s[0] * U[:, 0]))  # True
```

NumPy returns <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> as `Vt`, so its rows are the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mi>i</mi></msub></mrow></math></span>.

## Link to eigenvalues

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mo>=</mo><mi>V</mi><msup><mi mathvariant="normal">Σ</mi><mi mathvariant="normal">⊤</mi></msup><mi mathvariant="normal">Σ</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup><mo separator="true">,</mo><mspace width="2em"/><mi>A</mi><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo>=</mo><mi>U</mi><mi mathvariant="normal">Σ</mi><msup><mi mathvariant="normal">Σ</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>U</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span></span>
These are spectral decompositions (module 05), so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup></mrow></math></span> are the eigenvalues of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi></mrow></math></span>. That is how to find an SVD by hand. In code, never form <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi></mrow></math></span> for this: it squares the condition number (lesson 06.11).

## Rectangular matrices

For a <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>×</mo><mn>2</mn></mrow></math></span> matrix, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>×</mo><mn>2</mn></mrow></math></span> with two singular values; the third column of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> spans the left null space. `np.linalg.svd(A, full_matrices=False)` returns the thin version, which drops those unused columns.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>U</mi><mi mathvariant="normal">Σ</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> for every matrix: rotate, stretch along the axes, rotate; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><msub><mi>v</mi><mi>i</mi></msub><mo>=</mo><msub><mi>σ</mi><mi>i</mi></msub><msub><mi>u</mi><mi>i</mi></msub></mrow></math></span>.

</div>
