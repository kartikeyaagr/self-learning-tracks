---
id: "05.09"
course: "ml-maths-course"
module: "05"
slug: "when-diagonalisation-fails"
title: "When Diagonalisation Fails"
keyIdeas: ["A matrix with a repeated eigenvalue may have too few independent eigenvectors; it is then called defective.", "The shear ⟦845⟧ has eigenvalue 1 twice but only one eigenvector direction.", "Nearly defective matrices make ⟦1204⟧ badly conditioned, so ⟦1722⟧ is inaccurate."]
question: "Show that ⟦1721⟧ is not diagonalisable."
answer: "Step 1, ⟦1726⟧ is triangular, so both eigenvalues are 3. Step 2, eigenvectors solve ⟦1728⟧, i.e. ⟦1729⟧, so ⟦1725⟧. Step 3, only the direction ⟦310⟧: one eigenvector, not enough for a basis of ⟦662⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Spectral decomposition"]
tools: ["NumPy"]
topic: {"id": "05.02", "title": "Spectral decomposition"}
---

## The big picture

Diagonalisation needs a full set of eigenvectors. Usually you get one, but not always. Knowing the exceptions keeps you from trusting an eigendecomposition that is numerically meaningless.

## The shear

~~~dg-raw
%%viz f_plane x=-0.5,2.5 y=-0.5,1.5 tk=0.5 pg=0:0_1:0_1:1_0:1,m;0:0_1:0_2:1_1:1,a v=1,0,e₁,f | The shear [[1, 1], [0, 1]] slides the top of the square sideways. Only vectors along the x-axis (e₁, orange) keep their direction.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a matrix with too few eigenvectors**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>J</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> has eigenvalues 1 and 1.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>J</mi><mo>−</mo><mi>I</mi><mo stretchy="false">)</mo><mi>v</mi><mo>=</mo><mn>0</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span>: only the direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

One direction can't be a basis of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>J</mi></mrow></math></span> is not diagonalisable.

</div>

## In floating point

Exact defectiveness is rare in floating point, but near-defectiveness is common. The eigenvectors come out nearly parallel, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> is badly conditioned, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi><mi mathvariant="normal">Λ</mi><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> amplifies errors.

```
J = np.array([[1.0, 1], [0, 1 + 1e-10]])
vals, V = np.linalg.eig(J)
print(np.linalg.cond(V) > 1e9)  # True
```

## What to use instead

- **Jordan form** is the exact theory, but it is unstable to compute.
- **Schur form** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>Q</mi><mi>T</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> with orthogonal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> and triangular <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>T</mi></mrow></math></span> always exists and is stable; libraries use it.
- **SVD** (module 07) always works for any matrix.

## Where this breaks in practice

- **Repeated eigenvalues are fine for symmetric matrices.** The identity has eigenvalue 1 repeated <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> times and is trivially diagonal. Only non-symmetric matrices can be defective.

<div class="take">
<div class="lab">Key takeaway</div>

repeated eigenvalues can leave too few eigenvectors. Symmetric matrices never have this problem.

</div>
