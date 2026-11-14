---
id: "05.07"
course: "ml-maths-course"
module: "05"
slug: "diagonalisation"
title: "Diagonalisation"
keyIdeas: ["If ⟦8⟧ has ⟦233⟧ independent eigenvectors, ⟦1690⟧, with eigenvectors in ⟦1204⟧ and eigenvalues on the diagonal of ⟦1691⟧.", "Read it right to left: change to eigenvector coordinates, scale each coordinate, change back.", "Distinct eigenvalues guarantee independent eigenvectors."]
question: "Diagonalise ⟦1483⟧ and use it to compute ⟦1484⟧."
answer: "Step 1, eigenvectors ⟦111⟧ with ⟦1586⟧ and ⟦1352⟧ with ⟦1588⟧, so ⟦1696⟧, ⟦1697⟧. Step 2, coordinates of ⟦731⟧ in the eigenbasis: ⟦836⟧. Step 3, scale: ⟦1495⟧. Step 4, back: ⟦1698⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Spectral decomposition"]
tools: ["NumPy"]
topic: {"id": "05.02", "title": "Spectral decomposition"}
---

## The big picture

Eigenvectors make the best possible basis: in it, the matrix does nothing but multiply each coordinate by a number. Writing a matrix this way is called diagonalisation, and it turns hard questions into questions about a short list of numbers.

## The factorisation

Put the eigenvectors in the columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> and the eigenvalues on the diagonal of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Λ</mi></mrow></math></span> (capital lambda). The equations <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><msub><mi>v</mi><mi>i</mi></msub><mo>=</mo><msub><mi>λ</mi><mi>i</mi></msub><msub><mi>v</mi><mi>i</mi></msub></mrow></math></span>, all at once, say <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>V</mi><mo>=</mo><mi>V</mi><mi mathvariant="normal">Λ</mi></mrow></math></span>. If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> is invertible:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>=</mo><mi>V</mi><mi mathvariant="normal">Λ</mi><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span></span>
Read right to left:

1. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>: coordinates of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in the eigenvector basis.
2. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Λ</mi></mrow></math></span>: multiply each coordinate by its eigenvalue.
3. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span>: back to ordinary coordinates.

~~~dg-flow Diagonalisation as a pipeline.
x -(V⁻¹)-> eigen-coordinates -(Λ: scale each)-> scaled coordinates -(V)-> *Ax
~~~

```
A = np.array([[2.0, 1], [1, 2]])
V = np.array([[1.0, 1], [1, -1]]); L = np.diag([3.0, 1])
print(np.allclose(V @ L @ np.linalg.inv(V), A))  # True
```

## When it works

- If the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> eigenvalues are all different, the eigenvectors are independent and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is diagonalisable.
- Repeated eigenvalues may or may not give enough eigenvectors (lesson 05.09).
- Symmetric matrices always work (lesson 05.10).

<div class="ex">
<div class="lab">Worked example</div>

**a non-symmetric matrix**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> has eigenvalues 5 and 2 with eigenvectors <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>.

These are not perpendicular, but they are independent, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>V</mi><mi mathvariant="normal">Λ</mi><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>2</mn></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

</div>

```
A = np.array([[4.0, 1], [2, 3]]); V = np.array([[1.0, 1], [1, -2]])
print(np.round(np.linalg.inv(V) @ A @ V, 10))
```

<div class="take">
<div class="lab">Key takeaway</div>

with enough eigenvectors, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>V</mi><mi mathvariant="normal">Λ</mi><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>: in the eigenbasis, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> just scales.

</div>
