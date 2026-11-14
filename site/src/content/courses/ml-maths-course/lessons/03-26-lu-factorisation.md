---
id: "03.26"
course: "ml-maths-course"
module: "03"
slug: "lu-factorisation"
title: "LU Factorisation"
keyIdeas: ["Recording the multipliers of elimination gives ⟦943⟧: ⟦491⟧ lower triangular with ones on the diagonal, ⟦207⟧ upper triangular, ⟦481⟧ the row swaps.", "Solving ⟦816⟧ then takes a forward solve with ⟦491⟧ and a back solve with ⟦207⟧, each about ⟦583⟧ operations.", "Factor once in ⟦933⟧, then solve for many right-hand sides cheaply."]
question: "You must solve ⟦816⟧ for 1,000 different ⟦682⟧ with the same ⟦1092⟧ matrix ⟦8⟧. Compare the cost of 1,000 calls to `solve` with one LU factorisation followed by 1,000 triangular solves."
answer: "Step 1, each `solve` factors ⟦8⟧ again: about ⟦1110⟧ operations, so 1,000 calls cost about ⟦1111⟧. Step 2, one LU costs ⟦988⟧. Each pair of triangular solves costs about ⟦1112⟧, so 1,000 of them cost ⟦1113⟧. Step 3, total about ⟦1114⟧, roughly 770 times cheaper."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy", "SciPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

Elimination does the expensive work of reshaping <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, and then throws it away. If you write down the multipliers as you go, you get a factorisation of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> that can be reused for every new right-hand side.

## What LU is

While eliminating, put each multiplier <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>m</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></math></span> into position <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>i</mi><mo separator="true">,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></math></span> of a lower-triangular matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi></mrow></math></span> with 1s on its diagonal. What is left of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is the upper-triangular <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span>. Then
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mi>A</mi><mo>=</mo><mi>L</mi><mi>U</mi></mrow></math></span></span>
where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> records any row swaps.

<div class="ex">
<div class="lab">Worked example</div>

**LU of a 2 by 2**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>6</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>. Multiplier <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>6</mn><mi mathvariant="normal">/</mi><mn>2</mn><mo>=</mo><mn>3</mn></mrow></math></span>; R2 − 3·R1 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

Check: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mi>U</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>6</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mn>3</mn><mo>+</mo><mn>1</mn></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo>=</mo><mi>A</mi></mrow></math></span>.

</div>

## Solving with LU

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> becomes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mi>U</mi><mi>x</mi><mo>=</mo><mi>P</mi><mi>b</mi></mrow></math></span>. Set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo>=</mo><mi>U</mi><mi>x</mi></mrow></math></span>:

1. Solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mi>c</mi><mo>=</mo><mi>P</mi><mi>b</mi></mrow></math></span> top-down (forward substitution).
2. Solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi><mi>x</mi><mo>=</mo><mi>c</mi></mrow></math></span> bottom-up (back substitution).

```
import scipy.linalg
A = np.array([[2.0, 1, 1], [4, 3, 3], [8, 7, 9]])
P, L, U = scipy.linalg.lu(A)
print(np.allclose(P @ L @ U, A))  # True
lu, piv = scipy.linalg.lu_factor(A)
B = np.array([[4.0, 1], [10, 0], [24, 0]])
X = scipy.linalg.lu_solve((lu, piv), B)
print(np.allclose(A @ X, B))  # True
```

SciPy's `lu` returns <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> so that <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>P</mi><mi>L</mi><mi>U</mi></mrow></math></span>; it is the same factorisation with the permutation on the other side.

## Determinant for free

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mi>det</mi><mo>⁡</mo><mi>P</mi><mo>⋅</mo><mi>det</mi><mo>⁡</mo><mi>L</mi><mo>⋅</mo><mi>det</mi><mo>⁡</mo><mi>U</mi><mo>=</mo><mo>±</mo><msub><mo>∏</mo><mi>i</mi></msub><msub><mi>u</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub></mrow></math></span>, since <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>L</mi><mo>=</mo><mn>1</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>P</mi><mo>=</mo><mo>±</mo><mn>1</mn></mrow></math></span>.

## Cholesky, the symmetric cousin

For a symmetric positive definite matrix (like a covariance), <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>L</mi><msup><mi>L</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> with a single triangular factor. It costs half as much as LU and needs no pivoting. Gaussian processes and Gaussian likelihoods use it constantly.

<div class="take">
<div class="lab">Key takeaway</div>

LU is elimination with the multipliers saved. Factor once, then each new right-hand side costs only two triangular solves.

</div>
