---
id: "05.03"
course: "ml-maths-course"
module: "05"
slug: "finding-eigenvectors"
title: "Finding Eigenvectors"
keyIdeas: ["For each eigenvalue ⟦559⟧, the eigenvectors are the non-zero solutions of ⟦1603⟧.", "That solution set (with 0) is the eigenspace of ⟦559⟧: the null space of ⟦1602⟧.", "`np.linalg.eigh` returns unit-length eigenvectors as columns; check with ⟦1618⟧."]
question: "Find an eigenvector of ⟦1616⟧ for ⟦1617⟧ and for ⟦1613⟧."
answer: "Step 1, ⟦1617⟧: ⟦1625⟧. Both rows say ⟦1626⟧, so ⟦1627⟧. Check: ⟦1628⟧. Step 2, ⟦1613⟧: ⟦1629⟧, so ⟦1630⟧, giving ⟦1631⟧. Check: ⟦1632⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Invariant directions and spectra"]
tools: ["NumPy"]
topic: {"id": "05.01", "title": "Invariant directions and spectra"}
---

## The big picture

Once you have an eigenvalue, finding its eigenvectors is an old problem in disguise: solve a homogeneous linear system. Everything from module 04 about null spaces applies.

## The method

For each eigenvalue <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span>:

1. Form <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi></mrow></math></span> (subtract <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span> from each diagonal entry).
2. Solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi><mo stretchy="false">)</mo><mi>v</mi><mo>=</mo><mn>0</mn></mrow></math></span>. Because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi></mrow></math></span> is singular, there are non-zero solutions.
3. Any non-zero solution is an eigenvector.

The set of all solutions is the **eigenspace** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span>, a subspace.

<div class="ex">
<div class="lab">Worked example</div>

**eigenvector for λ = 1**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>−</mo><mi>I</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

Both rows say <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>1</mn></msub><mo>+</mo><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span>. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, or any multiple.

</div>

## In code

```
A = np.array([[2.0, 1], [1, 2]])
vals, vecs = np.linalg.eigh(A)
print(vals)  # [1. 3.]
print(np.allclose(A @ vecs, vecs * vals))  # True
```

`eigh` returns eigenvectors as the columns of `vecs`, length 1, eigenvalues sorted from small to large. `vecs * vals` scales column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mi>i</mi></msub></mrow></math></span>, so the check says <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><msub><mi>v</mi><mi>i</mi></msub><mo>=</mo><msub><mi>λ</mi><mi>i</mi></msub><msub><mi>v</mi><mi>i</mi></msub></mrow></math></span> for every column.

## Eigenspaces bigger than a line

An eigenvalue can have a whole plane of eigenvectors.

<div class="ex">
<div class="lab">Worked example</div>

**the identity**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>I</mi></mrow></math></span> (2 by 2) has eigenvalue 1 twice, and every non-zero vector is an eigenvector. The eigenspace is all of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mi mathvariant="normal">d</mi><mi mathvariant="normal">i</mi><mi mathvariant="normal">a</mi><mi mathvariant="normal">g</mi></mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span>: eigenvalue 2 has the eigenspace spanned by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>e</mi><mn>1</mn></msub></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>e</mi><mn>2</mn></msub></mrow></math></span>, a plane.

</div>

## Where this breaks in practice

- **`eig` vs `eigh`.** `np.linalg.eig` works for any square matrix but returns unsorted eigenvalues, possibly with tiny imaginary parts from rounding. For symmetric matrices use `eigh`.

<div class="take">
<div class="lab">Key takeaway</div>

eigenvectors for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span> are the null space of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi></mrow></math></span>.

</div>
