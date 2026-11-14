---
id: "05.01"
course: "ml-maths-course"
module: "05"
slug: "eigenvalues-and-eigenvectors-for-2x2-matrices"
title: "Eigenvalues and Eigenvectors for 2x2 Matrices"
keyIdeas: ["An eigenvector ⟦1578⟧ of a square matrix ⟦8⟧ satisfies ⟦1579⟧: ⟦8⟧ only stretches it, without turning it. ⟦559⟧ is the eigenvalue.", "Any non-zero multiple of an eigenvector is also one, so an eigenvector is really a direction.", "To test a candidate, multiply and check whether the result is a multiple of it."]
question: "For ⟦1483⟧, which of ⟦111⟧, ⟦1352⟧, ⟦310⟧ are eigenvectors, and with which eigenvalues?"
answer: "Step 1, ⟦1585⟧: eigenvector, ⟦1586⟧. Step 2, ⟦1587⟧: eigenvector, ⟦1588⟧. Step 3, ⟦1589⟧, which is not a multiple of ⟦310⟧: not an eigenvector."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Invariant directions and spectra"]
tools: ["NumPy"]
topic: {"id": "05.01", "title": "Invariant directions and spectra"}
---

## The big picture

A matrix usually both stretches and turns a vector. But most matrices have a few special directions where they **only stretch**. Those are the eigenvectors, and the stretch factors are the eigenvalues. Knowing them tells you what a matrix does when applied over and over, which explains exploding gradients, PageRank and slow optimisation.

## The definition

An **eigenvector** of a square matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is a non-zero vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> with
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mi>v</mi><mo>=</mo><mi>λ</mi><mi>v</mi></mrow></math></span></span>
Read it: "A times v is just a multiple of v". The number <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span> (Greek "lambda") is the **eigenvalue**. "Eigen" is German for "own": the matrix's own directions.

What different eigenvalues do to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span>:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>&gt;</mo><mn>1</mn></mrow></math></span>: stretches it.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>&lt;</mo><mi>λ</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span>: shrinks it.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>&lt;</mo><mn>0</mn></mrow></math></span>: flips it to point the other way.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>=</mo><mn>0</mn></mrow></math></span>: crushes it to zero, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> is in the null space.

~~~dg-raw
%%viz m_eig live | Drag x around the circle. Where Ax points the same way as x, you have found an eigenvector.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**test a few vectors**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo><mo>=</mo><mn>3</mn><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Same direction, 3 times longer. Eigenvector with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>=</mo><mn>3</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Unchanged. Eigenvector with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> points a different way. Not an eigenvector.

</div>

```
A = np.array([[2, 1], [1, 2]])
for v in ([1, 1], [1, -1], [1, 0]):
    print(v, A @ np.array(v))
```

## Directions, not vectors

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>v</mi><mo>=</mo><mi>λ</mi><mi>v</mi></mrow></math></span> then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mi>c</mi><mi>v</mi><mo stretchy="false">)</mo><mo>=</mo><mi>c</mi><mi>A</mi><mi>v</mi><mo>=</mo><mi>λ</mi><mo stretchy="false">(</mo><mi>c</mi><mi>v</mi><mo stretchy="false">)</mo></mrow></math></span>. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>5</mn><mo separator="true">,</mo><mo>−</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span> are eigenvectors too. Libraries return them scaled to length 1, with an arbitrary sign.

## A triangular shortcut

For a triangular matrix the eigenvalues are the diagonal entries. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span> has eigenvalues 4 and 2; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> is an eigenvector for 4 because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

an eigenvector is a direction the matrix only stretches; the eigenvalue is the stretch factor.

</div>
