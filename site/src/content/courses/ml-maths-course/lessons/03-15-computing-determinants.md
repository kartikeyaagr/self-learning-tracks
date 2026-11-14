---
id: "03.15"
course: "ml-maths-course"
module: "03"
slug: "computing-determinants"
title: "Computing Determinants"
keyIdeas: ["Cofactor expansion works by hand for ⟦804⟧ but needs about ⟦931⟧ operations in general.", "Software computes ⟦932⟧ from an LU factorisation, about ⟦933⟧ operations.", "Use `slogdet` (or Cholesky) for ⟦934⟧, because the determinant itself overflows or underflows."]
question: "⟦927⟧ in 300 dimensions. What is ⟦928⟧, what does float64 store, what is ⟦929⟧, and is ⟦930⟧ close to singular?"
answer: "Step 1, diagonal, so ⟦952⟧. Step 2, float64 can't go below about ⟦953⟧, so it stores 0.0. Step 3, ⟦954⟧, easy to store. Step 4, it is not nearly singular: every direction is shrunk by the same factor, so its condition number is 1, the best possible. Judge singularity with the condition number, not the determinant."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Determinants and singularity"]
tools: ["NumPy"]
topic: {"id": "03.02", "title": "Determinants and singularity"}
---

## The big picture

For a <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow></math></span> matrix, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mi>d</mi><mo>−</mo><mi>b</mi><mi>c</mi></mrow></math></span> is all you need. For bigger matrices there is a textbook formula that is hopeless in practice, and a practical method that comes from row reduction. Gaussian models need <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>det</mi><mo>⁡</mo></mrow></math></span> of large covariance matrices all the time, so the practical method matters.

## By hand for 3 by 3

**Cofactor expansion** along the first row:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>det</mi><mo>⁡</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>a</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>c</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>d</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>e</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>f</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>g</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>h</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>i</mi></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo>=</mo><mi>a</mi><mo stretchy="false">(</mo><mi>e</mi><mi>i</mi><mo>−</mo><mi>f</mi><mi>h</mi><mo stretchy="false">)</mo><mo>−</mo><mi>b</mi><mo stretchy="false">(</mo><mi>d</mi><mi>i</mi><mo>−</mo><mi>f</mi><mi>g</mi><mo stretchy="false">)</mo><mo>+</mo><mi>c</mi><mo stretchy="false">(</mo><mi>d</mi><mi>h</mi><mo>−</mo><mi>e</mi><mi>g</mi><mo stretchy="false">)</mo></mrow></math></span></span>
Each <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow></math></span> piece is the determinant of what remains after crossing out that entry's row and column. The signs alternate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>+</mo><mo separator="true">,</mo><mo>−</mo><mo separator="true">,</mo><mo>+</mo></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**a 3 by 3 determinant**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo stretchy="false">(</mo><mn>12</mn><mo>−</mo><mn>0</mn><mo stretchy="false">)</mo><mo>−</mo><mn>0</mn><mo>+</mo><mn>1</mn><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mn>24</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>25</mn></mrow></math></span>.

</div>

```
M = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 4]])
print(round(np.linalg.det(M), 6))  # 25.0
```

Cofactor expansion needs about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo stretchy="false">!</mo></mrow></math></span> operations. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>20</mn></mrow></math></span> that is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2.4</mn><mo>×</mo><mn>1</mn><msup><mn>0</mn><mn>18</mn></msup></mrow></math></span>.

## How software does it

Gaussian elimination (lessons 03.18 and 03.26) writes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mi>A</mi><mo>=</mo><mi>L</mi><mi>U</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> triangular, in about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>2</mn><mn>3</mn></mfrac><msup><mi>n</mi><mn>3</mn></msup></mrow></math></span> operations. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mo>±</mo><msub><mo>∏</mo><mi>i</mi></msub><msub><mi>u</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub></mrow></math></span>, with the sign set by the number of row swaps.

## Work with logs

The product of many numbers overflows or underflows. The log turns the product into a sum:

```
A = np.array([[4.0, 2], [2, 3]])
sign, logdet = np.linalg.slogdet(A)
print(sign, round(logdet, 4))  # 1.0 2.0794
L = np.linalg.cholesky(A)
print(round(2 * np.log(np.diag(L)).sum(), 4))  # 2.0794
```

Here <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>8</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ln</mi><mo>⁡</mo><mn>8</mn><mo>≈</mo><mn>2.0794</mn></mrow></math></span>. For a symmetric positive definite matrix like a covariance, Cholesky gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>2</mn><msub><mo>∑</mo><mi>i</mi></msub><mi>log</mi><mo>⁡</mo><msub><mi>L</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**Gaussian log-likelihood**

A Gaussian in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> dimensions has the term <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mi>log</mi><mo>⁡</mo><mi>det</mi><mo>⁡</mo><mi mathvariant="normal">Σ</mi></mrow></math></span> in its log density.

With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>=</mo><mn>768</mn></mrow></math></span> and eigenvalues around <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>2</mn></mrow></msup></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi mathvariant="normal">Σ</mi><mo>≈</mo><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>1536</mn></mrow></msup></mrow></math></span>, which is 0 in float64. `np.log(np.linalg.det(S))` returns `-inf`.

</div>

```
S = 1e-2 * np.eye(768)
print(np.linalg.det(S))  # 0.0
print(round(np.linalg.slogdet(S)[1]))  # -3537
```

## Where this breaks in practice

- **Tiny is not singular.** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>3</mn></mrow></msup><msub><mi>I</mi><mn>300</mn></msub><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>900</mn></mrow></msup></mrow></math></span>, but the matrix is perfectly well behaved. Use the condition number (lesson 03.31) to judge closeness to singular.

<div class="take">
<div class="lab">Key takeaway</div>

by hand, cofactors for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>×</mo><mn>3</mn></mrow></math></span>; in code, LU and `slogdet`. Never take the log of a determinant computed directly.

</div>
