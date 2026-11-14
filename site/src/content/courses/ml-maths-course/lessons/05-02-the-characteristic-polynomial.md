---
id: "05.02"
course: "ml-maths-course"
module: "05"
slug: "the-characteristic-polynomial"
title: "The Characteristic Polynomial"
keyIdeas: ["Eigenvalues solve ⟦1595⟧, the characteristic equation.", "For ⟦838⟧ it is ⟦1596⟧.", "Checks for any size: ⟦1597⟧ and ⟦1598⟧."]
question: "Find the eigenvalues of ⟦1594⟧ and check them against the trace and determinant."
answer: "Step 1, trace ⟦1609⟧, determinant ⟦1610⟧. Step 2, ⟦1611⟧ factors as ⟦1612⟧, so ⟦1613⟧ and ⟦132⟧. Step 3, check: ⟦1614⟧ and ⟦1615⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Invariant directions and spectra"]
tools: ["NumPy"]
topic: {"id": "05.01", "title": "Invariant directions and spectra"}
---

## The big picture

Guessing eigenvectors works for tidy examples only. There is a systematic way to find the eigenvalues first: they are exactly the numbers <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span> that make <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi></mrow></math></span> singular.

## Deriving the equation

Rewrite <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>v</mi><mo>=</mo><mi>λ</mi><mi>v</mi></mrow></math></span> as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi><mo stretchy="false">)</mo><mi>v</mi><mo>=</mo><mn>0</mn></mrow></math></span>. We need a **non-zero** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> that <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi></mrow></math></span> sends to zero. That happens exactly when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi></mrow></math></span> is singular (module 03):
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span></span>
This is the **characteristic equation**. The left side is a polynomial in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span> of degree <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>, the **characteristic polynomial**.

## The 2 by 2 case

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>det</mi><mo>⁡</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>a</mi><mo>−</mo><mi>λ</mi></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>c</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>d</mi><mo>−</mo><mi>λ</mi></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo>=</mo><msup><mi>λ</mi><mn>2</mn></msup><mo>−</mo><mo stretchy="false">(</mo><mi>a</mi><mo>+</mo><mi>d</mi><mo stretchy="false">)</mo><mi>λ</mi><mo>+</mo><mo stretchy="false">(</mo><mi>a</mi><mi>d</mi><mo>−</mo><mi>b</mi><mi>c</mi><mo stretchy="false">)</mo></mrow></math></span></span>
<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>+</mo><mi>d</mi></mrow></math></span> is the **trace** (sum of the diagonal) and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mi>d</mi><mo>−</mo><mi>b</mi><mi>c</mi></mrow></math></span> the determinant.

<div class="ex">
<div class="lab">Worked example</div>

**the full calculation**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>: trace 4, determinant 3.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>λ</mi><mn>2</mn></msup><mo>−</mo><mn>4</mn><mi>λ</mi><mo>+</mo><mn>3</mn><mo>=</mo><mo stretchy="false">(</mo><mi>λ</mi><mo>−</mo><mn>3</mn><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mi>λ</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Eigenvalues 3 and 1.

Checks: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>4</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>×</mo><mn>1</mn><mo>=</mo><mn>3</mn></mrow></math></span>.

</div>

```
A = np.array([[4.0, 1], [2, 3]])
print(np.poly(A))  # [ 1. -7. 10.]
print(np.sort(np.linalg.eigvals(A).real))  # [2. 5.]
```

`np.poly` returns the polynomial's coefficients, highest power first.

## Trace and determinant checks

For any size, the eigenvalues (counted with repetition, complex ones included) add up to the trace and multiply to the determinant:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">tr</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><munder><mo>∑</mo><mi>i</mi></munder><msub><mi>λ</mi><mi>i</mi></msub><mo separator="true">,</mo><mspace width="2em"/><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><munder><mo>∏</mo><mi>i</mi></munder><msub><mi>λ</mi><mi>i</mi></msub></mrow></math></span></span>
So a zero eigenvalue means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>0</mn></mrow></math></span>: singular.

## In practice

For bigger matrices nobody solves the polynomial; finding polynomial roots is numerically unstable. Libraries use the **QR algorithm**, about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>3</mn></msup></mrow></math></span> operations.

<div class="take">
<div class="lab">Key takeaway</div>

eigenvalues are the roots of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mo>−</mo><mi>λ</mi><mi>I</mi><mo stretchy="false">)</mo></mrow></math></span>. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow></math></span>, use trace and determinant.

</div>
