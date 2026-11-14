---
id: "05.10"
course: "ml-maths-course"
module: "05"
slug: "the-spectral-theorem"
title: "The Spectral Theorem"
keyIdeas: ["Every real symmetric matrix is ⟦1716⟧ with orthogonal ⟦571⟧ (⟦1731⟧) and real ⟦1691⟧.", "Its eigenvalues are real and eigenvectors for different eigenvalues are perpendicular.", "Geometrically: rotate, stretch along the axes, rotate back."]
question: "Show that eigenvectors of a symmetric ⟦8⟧ with different eigenvalues ⟦1730⟧ are perpendicular."
answer: "Step 1, let ⟦1737⟧, ⟦1738⟧. Step 2, ⟦1744⟧, using ⟦1645⟧. Step 3, so ⟦1745⟧, and since ⟦1730⟧, ⟦1740⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Spectral decomposition"]
tools: ["NumPy"]
topic: {"id": "05.02", "title": "Spectral decomposition"}
---

## The big picture

For a general matrix, diagonalisation can fail and eigenvalues can be complex. For symmetric matrices, none of that happens: you always get real eigenvalues and a perpendicular set of eigenvectors. Covariances, Hessians and kernel matrices are symmetric, so this theorem is behind PCA, second-order optimisation and Gaussian processes.

## The theorem

For real symmetric <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>=</mo><mi>Q</mi><mi mathvariant="normal">Λ</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mo separator="true">,</mo><mspace width="2em"/><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi><mo>=</mo><mi>I</mi><mo separator="true">,</mo><mspace width="1em"/><mi mathvariant="normal">Λ</mi><mtext> real</mtext></mrow></math></span></span>
The columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> are orthonormal eigenvectors, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Q</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> is simply <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>.

~~~dg-raw
%%viz m_spec live | A symmetric matrix is: turn, stretch along the axes, turn back. Press Play to see the three steps.
~~~

## Rotate, stretch, rotate back

Reading <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi><mi mathvariant="normal">Λ</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> right to left:

1. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>: rotate so the eigenvectors line up with the axes.
2. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Λ</mi></mrow></math></span>: stretch along each axis by its eigenvalue.
3. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span>: rotate back.

So a symmetric matrix turns a circle into an ellipse whose axes are the eigenvectors and whose half-lengths are <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><msub><mi>λ</mi><mi>i</mi></msub><mi mathvariant="normal">∣</mi></mrow></math></span>.

```
A = np.array([[2.0, 1], [1, 2]])
vals, Q = np.linalg.eigh(A)
print(np.allclose(Q.T @ Q, np.eye(2)), np.allclose(Q @ np.diag(vals) @ Q.T, A))  # True True
```

## Why eigenvectors are perpendicular

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>u</mi><mo>=</mo><mi>λ</mi><mi>u</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>v</mi><mo>=</mo><mi>μ</mi><mi>v</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo mathvariant="normal">≠</mo><mi>μ</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup><mi>v</mi><mo>=</mo><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>v</mi><mo>=</mo><mi>μ</mi><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup><mi>v</mi></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup><mi>v</mi><mo>=</mo><mn>0</mn></mrow></math></span>. (For repeated eigenvalues, perpendicular eigenvectors can still be chosen.)

<div class="ex">
<div class="lab">Worked example</div>

**a covariance matrix**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>: trace 7, determinant 8. Eigenvalues <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>7</mn><mo>±</mo><msqrt><mn>17</mn></msqrt><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>2</mn><mo>≈</mo><mn>5.56</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1.44</mn></mrow></math></span>.

The data spreads most along the first eigenvector and least along the second, and the two directions are perpendicular. That is the first step of PCA.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

symmetric matrices are rotate-stretch-rotate back, with real eigenvalues and perpendicular eigenvectors.

</div>
